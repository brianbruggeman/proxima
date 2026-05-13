//! HTTP/1.1 client — no hyper, no tokio.
//!
//! # What form is this?
//!
//! Everything in proxima that does work is a **pipe**: one async function,
//! `call`, that takes an `In` and returns a `Result<Out, Err>`. The trait is
//! [`Pipe`](proxima_primitives::pipe::Pipe), and its docs name the four
//! forms a pipe can take, chosen entirely by what you pick for `In` and
//! `Out`: a **transform** (`In -> Out`), a **source** (`() -> Out`), a
//! **sink** (`In -> ()`), and an **observe** (`In -> In`). There is only one
//! trait; the four names are what it looks like once the types are filled in.
//!
//! This client is a **transform**:
//!
//! ```text
//! Request<Bytes>  ──►  H1ClientUpstream  ──►  Response<Bytes>
//! ```
//!
//! Concretely, [`H1ClientUpstream`] sets
//! `In = `[`Request<Bytes>`](proxima_primitives::pipe::Request),
//! `Out = `[`Response<Bytes>`](proxima_primitives::pipe::Response), and
//! `Err = `[`ProximaError`]. A request goes in, a response comes out; that
//! is the whole contract. Nothing about "client" is a new concept — it is
//! the transform form with HTTP types plugged in.
//!
//! # Why `SendPipe` and not `Pipe`?
//!
//! The impl is of [`SendPipe`], not [`Pipe`](proxima_primitives::pipe::Pipe). These are the same shape;
//! `SendPipe` additionally promises the pipe and its future are `Send`, so
//! they can move between threads.
//!
//! The surprise is which one is the root. Most Rust async libraries require
//! `Send` everywhere. proxima does not: [`Pipe`](proxima_primitives::pipe::Pipe) has **no** `Send` bound and
//! [`SendPipe`] is the *additive* form. `Send`-everywhere is a work-stealing
//! assumption — it exists so a runtime can yank a task off one thread and
//! finish it on another. proxima's own runtime, prime, is per-core
//! shared-nothing: a task starts and ends on the core that owns it, so it
//! can hold an `Rc` or a `RefCell` and never pay for a bound it does not
//! use. This client opts INTO `Send` because it must also work on tokio,
//! which does steal work.
//!
//! # What it composes
//!
//! Three pieces, each traceable on its own:
//!
//! - **The transport** — [`StreamUpstream`], the primitive for "something
//!   that can hand me a byte stream." This client is generic over it, so it
//!   has no opinion about TCP vs. Unix sockets vs. TLS vs. a fake in a test.
//! - **The wire format** — the *sans-IO* codec in [`proxima_protocols`]:
//!   [`encode_request_head`], [`parse_response_head`], [`BodyDecoder`].
//!   "Sans-IO" means those functions never touch a socket: they take bytes
//!   and return parsed values, so they can be tested, fuzzed, and run
//!   anywhere. Splitting the protocol from the I/O is what lets the same
//!   HTTP/1.1 knowledge serve tokio, prime, DPDK, and a unit test.
//! - **The config** — [`H1ClientConfig`], the declarative half (host,
//!   label). See [`H1ClientUpstream::from_config`] for why the transport is
//!   not in it.
//!
//! # How it is built inside
//!
//! A transform on the outside; inside, it is the other three forms doing the
//! work. They are worth learning here because you can point at each one in
//! this one file:
//!
//! | stage | form | in this file |
//! |---|---|---|
//! | get a connection | **source** `() -> Conn` | [`StreamUpstream::poll_connect`] — nothing goes in, a connection comes out |
//! | apply per-request config | **observe** `In -> In` | `apply_config` — a `Request` goes in, the same `Request` comes out, adjusted |
//! | encode the head | **transform** `In -> Out` | [`encode_request_head`] — request fields in, bytes out |
//! | parse the head | **transform** `In -> Out` | [`parse_response_head`] — bytes in, a status + headers out |
//! | stream the body | **source** `() -> Out` | `body_stream` — pull it, get the next chunk |
//! | discard the body | **sink** `In -> ()` | `drain_body`, used by [`ResponseBodyMode::Drain`] — bytes in, nothing out |
//!
//! The four forms are not a taxonomy someone imposed afterwards. A client
//! genuinely needs a thing that produces connections from nothing, a thing
//! that adjusts a request without changing what it is, things that turn
//! values into bytes and back, and a thing that eats bytes and yields
//! nothing. Those are the four, and there are no others.
//!
//! ## Why these are not four `Pipe` impls chained with `AndThen`
//!
//! This is the part worth understanding, because the obvious question is why
//! the client is not written as
//! `connect.and_then(encode).and_then(parse)`. [`AndThen`](proxima_primitives::pipe::AndThen) — the composition
//! operator on [`Pipe`](proxima_primitives::pipe::Pipe) — passes a **value**
//! from one stage to the next: `First::Out` becomes `Second::In`, by
//! ownership.
//!
//! The stages here do not pass values. They share state:
//!
//! - the **connection** is cached across calls, so a second request reuses
//!   the socket instead of paying another TCP handshake (keep-alive);
//! - the **read and write buffers** are reused across calls, so the
//!   allocation is paid once per connection, not once per request;
//! - the parsed head **borrows** from the read buffer rather than copying out
//!   of it.
//!
//! Chaining them with `AndThen` would force each stage to own and hand off
//! what it produced, which is exactly what the buffer reuse exists to avoid.
//! So the stages are fused by hand into one `call`, and the whole fused thing
//! is the pipe. The forms describe the shape of each stage; `AndThen`
//! composes pipes whose stages are independent. Here they are not.
//!
//! That is the general rule, and it is worth carrying: **compose with
//! `AndThen` when stages are independent; fuse by hand when they share a
//! buffer or a connection.** A pipe is the boundary at which the work is
//! independent — not the smallest unit you can name.
//!
//! # Creating one
//!
//! [`H1ClientUpstream::from_config`] carries the worked example, end to end,
//! against an in-memory transport.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};
use proxima_primitives::sync::Mutex;
use tracing::{debug, warn};

use proxima_core::ProximaError;
use proxima_protocols::http1_codec::h1_body::{BodyDecoder, Status as BodyStatus};
use proxima_protocols::http1_codec::h1_client::{
    ResponseStatus, encode_request_head, framing_from_response, parse_response_head,
};
use proxima_primitives::pipe::body::ResponseStream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

pub use crate::http1::response_config::{
    ResponseBodyMode, ResponseHandling, ResponseHandlingConfig, ResponseHeaderMode,
};
use crate::http1::http_config::HttpUpstreamConfig;
use crate::templates::{TemplateContext, expand};

/// Read chunk size for draining the response off the connection. 16 KiB
/// balances syscall count against per-call buffer cost.
const READ_CHUNK_BYTES: usize = 16 * 1024;

/// Pre-allocation hint for the query string part of a request target:
/// one average key=value pair (algorithmic heuristic, not a hard limit).
const QUERY_EXTRA_CAPACITY: usize = 16;

/// Pre-allocation hint for the request head line + headers before the
/// body is appended. 64 bytes covers a minimal GET line + a few headers;
/// the Vec grows via doubling if the real head is larger.
const REQUEST_HEAD_CAPACITY_HINT: usize = 64;

// ── connection state ──────────────────────────────────────────────────────────

/// Reusable I/O buffers held per connection — cleared between requests so
/// allocations are paid once per connection, not once per request. The read
/// buffer absorbs the head + body; the write buffer holds the encoded request.
/// Both live behind the same `Mutex` as the connection itself.
struct ConnState<C: StreamConnection> {
    conn: Option<C>,
    read_buf: Vec<u8>,
    write_buf: Vec<u8>,
    /// The `(method, path, body)` the bytes in `write_buf` currently encode,
    /// for the no-header / no-query request shape. A repeated identical send
    /// (a load generator firing one cloned template, or any keep-alive client
    /// re-issuing the same call) skips the whole re-encode. The `Bytes` are
    /// held as live `Arc` clones, so a cache hit cannot collide with a freed-
    /// and-reused address (no ABA). Cleared whenever a richer request is sent.
    encoded: Option<(proxima_primitives::pipe::Method, Bytes, Bytes)>,
}

impl<C: StreamConnection> ConnState<C> {
    fn empty() -> Self {
        Self {
            conn: None,
            read_buf: Vec::with_capacity(READ_CHUNK_BYTES),
            write_buf: Vec::with_capacity(REQUEST_HEAD_CAPACITY_HINT),
            encoded: None,
        }
    }
}

// ── client struct ─────────────────────────────────────────────────────────────

/// HTTP/1.1 client. The **transform** form of
/// [`Pipe`](proxima_primitives::pipe::Pipe):
/// [`Request<Bytes>`](proxima_primitives::pipe::Request) in,
/// [`Response<Bytes>`](proxima_primitives::pipe::Response) out, failing with
/// [`ProximaError`]. See the [module docs](self) for what that means and why
/// the impl is [`SendPipe`].
///
/// One client owns one upstream binding (host:port, optionally TLS-wrapped)
/// and one cached connection reused across calls.
///
/// `U` is the transport: any [`StreamUpstream`]. The client never names a
/// socket type, which is why the same client works over
/// `proxima_net`'s TCP upstreams, `proxima_tls`'s TLS wrapper, or the
/// hand-written in-memory one in [`Self::from_config`]'s example.
///
/// Two ways to build one, both real:
///
/// - [`H1ClientUpstream::new`] — transport + host + label, positionally.
///   Fine when you are writing Rust and have all three in hand.
/// - [`H1ClientUpstream::from_config`] — an [`H1ClientConfig`] (which can
///   come from TOML or the environment) plus the transport. This is the path
///   that lets an operator move the host without a recompile.
pub struct H1ClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    // Same rationale as LengthPrefixedJsonClient::conn — interior
    // mutability for the `&self` Pipe API, holding the keep-alive
    // connection (pool of one) between calls. Single in-flight request
    // per client: the lock spans the head exchange, then the connection
    // moves into the streamed body and is returned here on clean EOF.
    // `Arc` so the streamed body (a `'static` stream) can hand the
    // connection back for keep-alive once it drains.
    state: Arc<Mutex<ConnState<U::Conn>>>,
    base_host: String,
    label: String,
    config: HttpUpstreamConfig,
    response: ResponseHandlingConfig,
}

impl<U: StreamUpstream> H1ClientUpstream<U> {
    /// `host` is the authority sent in the `Host` header (e.g.
    /// `"example.com"` or `"example.com:8443"`). `label` names the
    /// upstream for telemetry / `Pipe::name`.
    pub fn new(upstream: U, host: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            upstream: Arc::new(upstream),
            state: Arc::new(Mutex::new(ConnState::empty())),
            base_host: host.into(),
            label: label.into(),
            config: HttpUpstreamConfig::default(),
            response: ResponseHandlingConfig::default(),
        }
    }

    /// Attach an [`HttpUpstreamConfig`] so this client honors the same
    /// per-request knobs as the hyper-backed `HttpUpstream`: method
    /// override, the `forward_request_headers` allow-list, injected
    /// (template-expanded) headers, and the per-request `timeout`.
    #[must_use]
    pub fn with_config(mut self, config: HttpUpstreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the response-handling composition by name (preset).
    #[must_use]
    pub fn with_response(mut self, preset: ResponseHandling) -> Self {
        self.response = ResponseHandlingConfig::from_preset(preset);
        self
    }

    /// Set the response-handling composition granularly.
    #[must_use]
    pub fn with_response_config(mut self, response: ResponseHandlingConfig) -> Self {
        self.response = response;
        self
    }

    /// Set the response body mode independently.
    #[must_use]
    pub fn with_response_body(mut self, body: ResponseBodyMode) -> Self {
        self.response.body = body;
        self
    }

    /// Set the response header mode independently.
    #[must_use]
    pub fn with_response_headers(mut self, headers: ResponseHeaderMode) -> Self {
        self.response.headers = headers;
        self
    }

    /// Build from an [`H1ClientConfig`] (the declarative half: host +
    /// label) plus the live `upstream` transport (the runtime half).
    ///
    /// # Why two arguments and not one config
    ///
    /// A config is data — text you can write in a TOML file, ship over a
    /// network, or read from an environment variable. A transport is a live
    /// object holding an open socket. The second cannot be spelled in TOML,
    /// so it is injected here instead of pretending to be configuration.
    /// [`proxima_telemetry`]'s `Recorder::from_config` splits the same way,
    /// for the same reason.
    ///
    /// The payoff: an operator changes the host in a config file, and no
    /// Rust changes — because the config layer never had to name a socket
    /// type.
    ///
    /// # Making one, end to end
    ///
    /// This example is compiled and run by `cargo test`, so it cannot drift
    /// away from the API. It builds the config fluently, injects a transport,
    /// and drives one real request/response exchange.
    ///
    /// The transport here is hand-written and in-memory: it replays canned
    /// response bytes and throws away what is written to it. That is the
    /// point of the client being generic over [`StreamUpstream`] — no socket,
    /// no port, no server, and the HTTP/1.1 machinery is entirely real.
    ///
    /// ```
    /// use std::io;
    /// use std::pin::Pin;
    /// use std::task::{Context, Poll};
    ///
    /// use bytes::Bytes;
    /// use futures::io::{AsyncRead, AsyncWrite};
    /// use proxima_http::http1::{H1ClientConfig, H1ClientUpstream};
    /// use proxima_primitives::pipe::{Request, SendPipe};
    /// use proxima_primitives::stream::{PeerInfo, StreamConnection, StreamUpstream};
    ///
    /// // ── the fake transport ────────────────────────────────────────────
    /// // A `StreamConnection` is just "bytes in, bytes out, and it can name
    /// // its peer". Reads hand back a canned HTTP/1.1 response; writes are
    /// // accepted and dropped.
    /// struct CannedConnection {
    ///     response: &'static [u8],
    ///     sent: usize,
    /// }
    ///
    /// impl AsyncRead for CannedConnection {
    ///     fn poll_read(
    ///         mut self: Pin<&mut Self>,
    ///         _cx: &mut Context<'_>,
    ///         out: &mut [u8],
    ///     ) -> Poll<io::Result<usize>> {
    ///         let remaining = &self.response[self.sent..];
    ///         let count = remaining.len().min(out.len());
    ///         out[..count].copy_from_slice(&remaining[..count]);
    ///         self.sent += count;
    ///         // count == 0 is EOF, which is what ends the response body.
    ///         Poll::Ready(Ok(count))
    ///     }
    /// }
    ///
    /// impl AsyncWrite for CannedConnection {
    ///     fn poll_write(
    ///         self: Pin<&mut Self>,
    ///         _cx: &mut Context<'_>,
    ///         data: &[u8],
    ///     ) -> Poll<io::Result<usize>> {
    ///         Poll::Ready(Ok(data.len()))
    ///     }
    ///     fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
    ///         Poll::Ready(Ok(()))
    ///     }
    ///     fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
    ///         Poll::Ready(Ok(()))
    ///     }
    /// }
    ///
    /// impl StreamConnection for CannedConnection {
    ///     fn peer(&self) -> Option<PeerInfo> {
    ///         None
    ///     }
    /// }
    ///
    /// // A `StreamUpstream` is "something that can hand me a connection".
    /// struct CannedUpstream;
    ///
    /// impl StreamUpstream for CannedUpstream {
    ///     type Conn = CannedConnection;
    ///     fn poll_connect(&self, _cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
    ///         Poll::Ready(Ok(CannedConnection {
    ///             response: b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello",
    ///             sent: 0,
    ///         }))
    ///     }
    /// }
    ///
    /// // ── creation ──────────────────────────────────────────────────────
    /// // 1. the declarative half, via the fluent builder. `host` is what
    /// //    lands in the `Host:` header; `label` names it in telemetry.
    /// let config = H1ClientConfig::builder()
    ///     .host("example.com".to_string())
    ///     .label("demo".to_string())
    ///     .build();
    ///
    /// // 2. config + live transport -> a working transform.
    /// let client = H1ClientUpstream::from_config(CannedUpstream, &config);
    ///
    /// // ── use ───────────────────────────────────────────────────────────
    /// // `call` is the one method on the pipe. Request in, Response out.
    /// let request = Request::builder().method("GET").path("/").build().unwrap();
    /// let response = futures::executor::block_on(client.call(request)).unwrap();
    ///
    /// assert_eq!(response.status, 200);
    ///
    /// // the body arrives as a stream, so a large response is never held
    /// // whole in memory. `collect_body` is the "I'll take all of it" case.
    /// let body = futures::executor::block_on(response.collect_body()).unwrap();
    /// assert_eq!(&body[..], b"hello");
    /// ```
    pub fn from_config(upstream: U, config: &H1ClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            state: Arc::new(Mutex::new(ConnState::empty())),
            base_host: config.host.clone(),
            label: config.label.clone(),
            config: HttpUpstreamConfig::default(),
            response: config.response,
        }
    }

    /// Fluent builder for the declarative half. Set `host` (and
    /// optionally `label`), then pair it with a live transport via
    /// [`Self::from_config`].
    pub fn config_builder() -> H1ClientConfigBuilder {
        H1ClientConfig::builder()
    }

    /// Expose the resolved response-handling config. Used by the parity test
    /// to assert config-path and builder-path produce identical state.
    #[must_use]
    pub fn response_config(&self) -> ResponseHandlingConfig {
        self.response
    }
}

fn default_label() -> String {
    "h1-client".to_string()
}

/// HTTP status → coarse class label for upstream telemetry. Mirrors the
/// hyper `HttpUpstream` so the prime backend emits identical metric labels.
fn status_class(status: u16) -> &'static str {
    match status {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
}

/// The declarative, serializable description of an [`H1ClientUpstream`] —
/// the half that can live in env / TOML. The live transport (a
/// [`StreamUpstream`]) is *not* here: it's a runtime object injected at
/// [`H1ClientUpstream::from_config`] time, mirroring telemetry's
/// config-vs-runtime split.
///
/// # Three ways in, one type out
///
/// The derives above are not decoration; each one buys a real entry point,
/// and all three produce the same `H1ClientConfig`:
///
/// | source | how | notes |
/// |---|---|---|
/// | code | `H1ClientConfig::builder()` | from `bon`'s `Builder` |
/// | a file | `conflaguration::from_file(path)` | TOML or JSON, from `Deserialize` |
/// | the environment | `H1ClientConfig::from_env()` | from `conflaguration`'s `Settings` |
///
/// The env keys are the `H1_CLIENT` prefix plus the field name, uppercased:
/// `H1_CLIENT_HOST` and `H1_CLIENT_LABEL`. `host` has no default, so
/// [`from_env`](conflaguration::Settings::from_env) with `H1_CLIENT_HOST`
/// unset fails rather than inventing a host to talk to.
///
/// To layer sources rather than pick one, build a config from any source and
/// then call
/// [`override_from_env`](conflaguration::Settings::override_from_env): it
/// overwrites only the fields whose env keys are actually set, leaving the
/// rest alone.
///
/// # Building one
///
/// Compiled and run by `cargo test`. Both paths, plus the layering:
///
/// ```
/// use conflaguration::{Settings, Validate};
/// use proxima_http::http1::H1ClientConfig;
///
/// // ── from code, fluently ───────────────────────────────────────────────
/// let built = H1ClientConfig::builder()
///     .host("huggingface.co".to_string())
///     .label("hf".to_string())
///     .build();
/// assert_eq!(built.host, "huggingface.co");
///
/// // `label` has a default, so the fluent surface can omit it.
/// let defaulted = H1ClientConfig::builder().host("example.com".to_string()).build();
/// assert_eq!(defaulted.label, "h1-client");
///
/// // Validate is a separate step, on purpose: constructing a config and
/// // deciding it is sane are different jobs, and only the second one
/// // has an opinion.
/// assert!(built.validate().is_ok());
/// let empty_host = H1ClientConfig::builder().host(String::new()).build();
/// assert!(empty_host.validate().is_err(), "an empty host is rejected");
///
/// // ── from a file ───────────────────────────────────────────────────────
/// let dir = tempfile::TempDir::new().unwrap();
/// let path = dir.path().join("client.toml");
/// std::fs::write(&path, "host = \"huggingface.co\"\nlabel = \"hf\"\n").unwrap();
///
/// let loaded: H1ClientConfig = conflaguration::from_file(&path).unwrap();
/// assert_eq!(loaded.host, "huggingface.co");
/// assert_eq!(loaded.label, "hf");
///
/// // a file may set only some fields; the rest fall back to their defaults.
/// let partial_path = dir.path().join("partial.toml");
/// std::fs::write(&partial_path, "host = \"example.com\"\n").unwrap();
/// let partial: H1ClientConfig = conflaguration::from_file(&partial_path).unwrap();
/// assert_eq!(partial.label, "h1-client", "not set in the file, so the default");
///
/// // ── layering env on top ───────────────────────────────────────────────
/// // Overwrites only the fields whose H1_CLIENT_* keys are set. Nothing is
/// // set in this test process, so the file's values survive untouched —
/// // which is exactly the contract being demonstrated.
/// let mut layered = loaded;
/// layered.override_from_env().unwrap();
/// assert_eq!(layered.host, "huggingface.co");
/// ```
///
/// Pair the result with a transport via [`H1ClientUpstream::from_config`],
/// which is where the worked end-to-end example lives.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "H1_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct H1ClientConfig {
    /// Authority sent in the `Host` header, e.g. `"huggingface.co"` or
    /// `"example.com:8443"`. Required.
    pub host: String,
    /// Name for telemetry / `Pipe::name`. Defaults to `"h1-client"`.
    #[setting(default = "h1-client")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub label: String,
    /// Response-handling composition: body mode (collect/drain) and header
    /// mode (all/framing). Defaults to the full `Collect+All` path — identical
    /// to the previous behaviour. Set to `Drain+Framing` for load generators.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub response: ResponseHandlingConfig,
}

impl Validate for H1ClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.label.is_empty() {
            errors.push(ValidationMessage::new("label", "must be non-empty"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

// ── request encoding ──────────────────────────────────────────────────────────

/// Build the request-target (path + `?query`) the way an origin server
/// expects it on the request line.
fn request_target(request: &Request<Bytes>) -> String {
    let path = String::from_utf8_lossy(request.path.as_ref());
    if request.query.is_empty() {
        return path.into_owned();
    }
    let mut target = String::with_capacity(path.len() + QUERY_EXTRA_CAPACITY);
    target.push_str(&path);
    let mut first = true;
    for (name, value) in &request.query {
        target.push(if first { '?' } else { '&' });
        first = false;
        target.push_str(&String::from_utf8_lossy(name));
        target.push('=');
        target.push_str(&String::from_utf8_lossy(value));
    }
    target
}

/// Serialize the request head + buffered body into one buffer. Adds
/// `Host`, `Content-Length`, and `Connection: keep-alive` when the
/// caller hasn't already supplied them — case-insensitively.
fn encode_request(request: &Request<Bytes>, base_host: &str, body: &Bytes, out: &mut Vec<u8>) {
    out.clear();
    let mut header_pairs: Vec<(Bytes, Bytes)> = request
        .metadata
        .iter()
        .map(|(name, value)| (Bytes::clone(name), Bytes::clone(value)))
        .collect();
    if !request.metadata.contains_key("host") {
        header_pairs.push((
            Bytes::from_static(b"host"),
            Bytes::copy_from_slice(base_host.as_bytes()),
        ));
    }
    if !request.metadata.contains_key("content-length")
        && !request.metadata.contains_key("transfer-encoding")
    {
        let length = body.len().to_string();
        header_pairs.push((Bytes::from_static(b"content-length"), Bytes::from(length)));
    }
    if !request.metadata.contains_key("connection") {
        header_pairs.push((
            Bytes::from_static(b"connection"),
            Bytes::from_static(b"keep-alive"),
        ));
    }
    let method = String::from_utf8_lossy(request.method.as_bytes());
    let target = request_target(request);
    out.reserve(REQUEST_HEAD_CAPACITY_HINT + body.len());
    encode_request_head(&method, &target, &header_pairs, out);
    out.extend_from_slice(body);
}

// ── response head reading ─────────────────────────────────────────────────────

/// Which headers to collect depends on the mode: `All` copies every header;
/// `Framing` only copies the three the client itself needs to advance the
/// connection state.
#[inline]
fn is_framing_header(name: &[u8]) -> bool {
    name.eq_ignore_ascii_case(b"content-length")
        || name.eq_ignore_ascii_case(b"transfer-encoding")
        || name.eq_ignore_ascii_case(b"connection")
}

/// Read off `conn` until the response head parses Complete. Returns the
/// status, header pairs (filtered by `header_mode`), the chosen body
/// framing, and the offset in `read_buf` where the body begins (the head
/// leftover the caller feeds the body decoder — borrowed, not copied).
async fn read_head<C: StreamConnection>(
    conn: &mut C,
    read_buf: &mut Vec<u8>,
    header_mode: ResponseHeaderMode,
) -> Result<
    (
        u16,
        Vec<(Bytes, Bytes)>,
        proxima_protocols::http1_codec::h1_body::BodyFraming,
        usize,
    ),
    ProximaError,
> {
    read_buf.clear();
    let mut scratch = [0_u8; READ_CHUNK_BYTES];
    loop {
        let read = conn
            .read(&mut scratch)
            .await
            .map_err(|err| ProximaError::Upstream(format!("read response head: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "connection closed before response head".into(),
            ));
        }
        read_buf.extend_from_slice(&scratch[..read]);
        match parse_response_head(read_buf)
            .map_err(|err| ProximaError::Upstream(format!("parse response head: {err}")))?
        {
            ResponseStatus::Partial => {}
            ResponseStatus::Complete { head, body_offset } => {
                let status = head.status;
                let framing = framing_from_response(&head);
                let header_pairs = head
                    .headers
                    .iter()
                    .filter(|header| match header_mode {
                        ResponseHeaderMode::All => true,
                        ResponseHeaderMode::Framing => is_framing_header(header.name()),
                    })
                    .map(|header| {
                        (
                            Bytes::copy_from_slice(header.name()),
                            Bytes::copy_from_slice(header.value()),
                        )
                    })
                    .collect();
                return Ok((status, header_pairs, framing, body_offset));
            }
        }
    }
}

/// Read the response head and return ONLY the status, body framing, and the
/// body offset in `read_buf` — no header pairs are copied. The `send_raw`
/// fast path uses this: a load generator / liveness prober needs the status
/// and the keep-alive boundary, never the header values, so the per-response
/// `Vec<(Bytes, Bytes)>` + the per-header `Bytes::copy_from_slice` are skipped.
async fn read_head_status<C: StreamConnection>(
    conn: &mut C,
    read_buf: &mut Vec<u8>,
) -> Result<(u16, proxima_protocols::http1_codec::h1_body::BodyFraming, usize), ProximaError> {
    read_buf.clear();
    let mut scratch = [0_u8; READ_CHUNK_BYTES];
    loop {
        let read = conn
            .read(&mut scratch)
            .await
            .map_err(|err| ProximaError::Upstream(format!("read response head: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "connection closed before response head".into(),
            ));
        }
        read_buf.extend_from_slice(&scratch[..read]);
        match parse_response_head(read_buf)
            .map_err(|err| ProximaError::Upstream(format!("parse response head: {err}")))?
        {
            ResponseStatus::Partial => {}
            ResponseStatus::Complete { head, body_offset } => {
                return Ok((head.status, framing_from_response(&head), body_offset));
            }
        }
    }
}

// ── body streaming ────────────────────────────────────────────────────────────

/// State the lazy body stream carries between polls: the live connection
/// (taken out of the pool), the in-progress decoder, the not-yet-fed
/// head leftover, and an `Arc` handle to return the connection to the
/// pool for keep-alive once the body drains cleanly.
struct BodyPump<C: StreamConnection> {
    conn: Option<C>,
    decoder: BodyDecoder,
    seed: Option<Vec<u8>>,
    pool: Arc<Mutex<ConnState<C>>>,
    finished: bool,
}

/// Build the lazy response-body stream. Decodes the body off `conn`
/// incrementally — each socket read that yields decoded bytes becomes one
/// stream chunk, so an SSE/`text/event-stream` upstream is forwarded
/// token-by-token instead of buffered whole. On clean end-of-body the
/// connection is returned to `pool` for keep-alive; any decode/read error
/// drops it (the next call reconnects). Byte-output parity with the prior
/// buffered `read_body` is asserted by `streamed_body_matches_buffered`.
fn body_stream<C: StreamConnection>(
    conn: C,
    framing: proxima_protocols::http1_codec::h1_body::BodyFraming,
    seed: Vec<u8>,
    pool: Arc<Mutex<ConnState<C>>>,
) -> impl futures::stream::Stream<Item = Result<Bytes, ProximaError>> + Send + 'static {
    let pump = BodyPump {
        conn: Some(conn),
        decoder: BodyDecoder::new(framing),
        seed: Some(seed),
        pool,
        finished: false,
    };
    futures::stream::unfold(pump, |mut pump| async move {
        if pump.finished {
            return None;
        }
        let mut out: Vec<u8> = Vec::new();
        let mut scratch = [0_u8; READ_CHUNK_BYTES];
        loop {
            // drain the head leftover before touching the socket.
            if let Some(seed) = pump.seed.take() {
                match pump
                    .decoder
                    .feed(&seed, |chunk| out.extend_from_slice(chunk))
                {
                    Ok((_consumed, BodyStatus::End)) => {
                        return finish(pump, out).await;
                    }
                    Ok((_consumed, BodyStatus::NeedMore)) => {}
                    Err(err) => return fail(pump, format!("decode body: {err}")),
                }
                if !out.is_empty() {
                    return Some((Ok(Bytes::from(out)), pump));
                }
            }
            let read = match pump.conn.as_mut() {
                Some(conn) => match conn.read(&mut scratch).await {
                    Ok(read) => read,
                    Err(err) => return fail(pump, format!("read response body: {err}")),
                },
                None => return None,
            };
            if read == 0 {
                return fail(pump, "connection closed mid response body".to_string());
            }
            match pump
                .decoder
                .feed(&scratch[..read], |chunk| out.extend_from_slice(chunk))
            {
                Ok((_consumed, BodyStatus::End)) => return finish(pump, out).await,
                Ok((_consumed, BodyStatus::NeedMore)) => {
                    if !out.is_empty() {
                        return Some((Ok(Bytes::from(out)), pump));
                    }
                }
                Err(err) => return fail(pump, format!("decode body: {err}")),
            }
        }
    })
}

/// Consume a response body to the keep-alive boundary WITHOUT materializing
/// it or building the lazy stream — the `Drain` fast path. Feeds the head
/// leftover then reads the socket into a stack scratch buffer, discarding all
/// decoded output, until the framing decoder reports `End`. The connection is
/// left untouched in the pool slot (caller keeps it on `Ok`, drops it on
/// `Err`). This is the zero-alloc analog of `body_stream(..).drain()`.
async fn drain_body<C: StreamConnection>(
    conn: &mut C,
    framing: proxima_protocols::http1_codec::h1_body::BodyFraming,
    seed: &[u8],
) -> Result<(), ProximaError> {
    let mut decoder = BodyDecoder::new(framing);
    let (_, status) = decoder
        .feed(seed, |_| {})
        .map_err(|err| ProximaError::Upstream(format!("decode body: {err}")))?;
    if matches!(status, BodyStatus::End) {
        return Ok(());
    }
    let mut scratch = [0_u8; READ_CHUNK_BYTES];
    loop {
        let read = conn
            .read(&mut scratch)
            .await
            .map_err(|err| ProximaError::Upstream(format!("read response body: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "connection closed mid response body".into(),
            ));
        }
        let (_, status) = decoder
            .feed(&scratch[..read], |_| {})
            .map_err(|err| ProximaError::Upstream(format!("decode body: {err}")))?;
        if matches!(status, BodyStatus::End) {
            return Ok(());
        }
    }
}

/// Terminal step: hand the connection back to the pool for keep-alive,
/// then yield any final bytes (or end the stream when there are none).
async fn finish<C: StreamConnection>(
    mut pump: BodyPump<C>,
    out: Vec<u8>,
) -> Option<(Result<Bytes, ProximaError>, BodyPump<C>)> {
    if let Some(conn) = pump.conn.take() {
        let mut guard = pump.pool.lock().await;
        guard.conn = Some(conn);
    }
    pump.finished = true;
    if out.is_empty() {
        None
    } else {
        Some((Ok(Bytes::from(out)), pump))
    }
}

/// Error step: drop the connection (its framing state is now unknown so
/// it cannot be reused) and yield the error as the stream's last item.
fn fail<C: StreamConnection>(
    mut pump: BodyPump<C>,
    message: String,
) -> Option<(Result<Bytes, ProximaError>, BodyPump<C>)> {
    pump.conn = None;
    pump.finished = true;
    Some((Err(ProximaError::Upstream(message)), pump))
}

// ── config application ────────────────────────────────────────────────────────

/// Apply the [`HttpUpstreamConfig`] to the outbound request, mirroring
/// the hyper-backed `HttpUpstream::call` exactly: method override, then
/// the `forward_request_headers` allow-list (case-insensitive linear
/// scan), then the template-expanded injected headers. `None` allow-list
/// forwards every header.
fn apply_config(request: &mut Request<Bytes>, config: &HttpUpstreamConfig) {
    if let Some(override_method) = &config.method_override {
        request.method = proxima_primitives::pipe::Method::from_bytes(override_method.as_bytes());
    }
    if let Some(allowed) = &config.forward_request_headers {
        request.metadata.retain(|name, _| {
            allowed
                .iter()
                .any(|allowed_name| allowed_name.as_ref().eq_ignore_ascii_case(name.as_ref()))
        });
    }
    if !config.injected_request_headers.is_empty() {
        let trace_id_str = request
            .context
            .trace_id
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
        let pipe_label_str = request
            .context
            .pipe_label
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
        let template_context = TemplateContext {
            request_id: None,
            trace_id: trace_id_str,
            pipe: pipe_label_str,
            body: None,
        };
        for (name, value) in &config.injected_request_headers {
            let expanded = expand(value, &template_context);
            request.metadata.insert(name.clone(), expanded);
        }
    }
}

// ── Pipe impl ─────────────────────────────────────────────────────────────────

impl<U: StreamUpstream> SendPipe for H1ClientUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let timeout = self.config.timeout;
            let exchange = self.exchange(request);
            match timeout {
                Some(limit) => match proxima_core::time::timeout(limit, exchange).await {
                    Ok(result) => result,
                    Err(_) => Err(ProximaError::Timeout(limit)),
                },
                None => exchange.await,
            }
        }
    }
}


impl<U: StreamUpstream> H1ClientUpstream<U> {
    /// Drive one request over the cached connection up to the response
    /// head, then return a `Response` whose body streams lazily off the
    /// connection. Wrapped by `call` in the per-request `timeout` race —
    /// note the timeout now bounds connect + request + response-head
    /// (time-to-first-byte), NOT the full body: a streamed body has no
    /// single deadline, which is the correct shape for an SSE upstream.
    async fn exchange(&self, mut request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        apply_config(&mut request, &self.config);
        let (request, body) = request.body_bytes().await?;

        let telemetry = request.context.telemetry.clone();
        let started = Instant::now();

        let mut guard = self.state.lock().await;
        let reused = guard.conn.is_some();
        if guard.conn.is_none() {
            let conn = self
                .upstream
                .connect()
                .await
                .map_err(|err| ProximaError::Upstream(format!("connect: {err}")))?;
            debug!(label = %self.label, "h1 client connected");
            guard.conn = Some(conn);
        }

        // encode into the reusable write buffer, then write + read head. A
        // request with no extra headers and no query encodes to bytes that
        // depend only on (method, path, body) + the fixed host — so a repeat of
        // the same call reuses the buffer verbatim and skips the re-encode (the
        // header_pairs Vec, the content-length format, the head serialization).
        let cacheable = request.query.is_empty() && request.metadata.is_empty();
        let cache_hit = cacheable
            && guard
                .encoded
                .as_ref()
                .is_some_and(|(method, path, encoded_body)| {
                    *method == request.method && *path == request.path && *encoded_body == body
                });
        if !cache_hit {
            encode_request(&request, &self.base_host, &body, &mut guard.write_buf);
            guard.encoded =
                cacheable.then(|| (request.method.clone(), request.path.clone(), body.clone()));
        }

        let response_mode = self.response;
        let first = Self::write_and_read_head(&mut guard, response_mode.headers).await;
        let outcome = match first {
            Ok(value) => Ok(value),
            Err(error) if reused => {
                warn!(label = %self.label, error = %error, "reused h1 connection failed; reconnecting and retrying once");
                guard.conn = None;
                let conn = self
                    .upstream
                    .connect()
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("reconnect: {err}")))?;
                guard.conn = Some(conn);
                Self::write_and_read_head(&mut guard, response_mode.headers).await
            }
            Err(error) => Err(error),
        };

        match outcome {
            Ok((status, header_pairs, framing, body_offset)) => {
                if telemetry.is_active() {
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                    let labels = request.context.metric_labels(&[
                        ("upstream", self.label.as_str()),
                        ("status_class", status_class(status)),
                    ]);
                    telemetry.counter_inc("proxima.upstream.calls_total", &labels, 1);
                    telemetry.histogram_record("proxima.upstream.latency_ms", &labels, elapsed_ms);
                }

                let mut response = Response::new(status);
                for (name, value) in header_pairs {
                    response.metadata.insert(name, value);
                }
                match response_mode.body {
                    ResponseBodyMode::Drain => {
                        // consume the body inline to the keep-alive boundary — no
                        // streaming machinery, no per-chunk alloc, and the head
                        // leftover is fed as a BORROW of read_buf (no copy). on clean
                        // end the connection stays in the pool slot; an error drops it.
                        let result = {
                            let ConnState { conn, read_buf, .. } = &mut *guard;
                            match conn.as_mut() {
                                Some(conn) => {
                                    drain_body(conn, framing, &read_buf[body_offset..]).await
                                }
                                None => Err(ProximaError::Upstream(
                                    "connection slot empty after head".into(),
                                )),
                            }
                        };
                        match result {
                            Ok(()) => {
                                drop(guard);
                                Ok(response)
                            }
                            Err(err) => {
                                guard.conn = None;
                                drop(guard);
                                Err(err)
                            }
                        }
                    }
                    ResponseBodyMode::Collect => {
                        // collect's lazy stream is `'static` (the connection moves
                        // into it), so it needs an owned seed — the one to_vec the
                        // Drain fast path avoids.
                        let seed = guard.read_buf[body_offset..].to_vec();
                        let conn = guard.conn.take().ok_or_else(|| {
                            ProximaError::Upstream("connection slot empty after head".into())
                        })?;
                        drop(guard);
                        let stream = body_stream(conn, framing, seed, Arc::clone(&self.state));
                        Ok(response.with_stream(ResponseStream::new(stream)))
                    }
                }
            }
            Err(error) => {
                if telemetry.is_active() {
                    let labels = request.context.metric_labels(&[
                        ("upstream", self.label.as_str()),
                        ("status_class", "error"),
                    ]);
                    telemetry.counter_inc("proxima.upstream.errors_total", &labels, 1);
                }
                guard.conn = None;
                warn!(label = %self.label, error = %error, "h1 client head exchange failed, dropping connection");
                Err(error)
            }
        }
    }

    /// Write the request and read just the response head over the
    /// connection currently in `guard` (which the caller has ensured is
    /// `Some`). Returns the status, header pairs, body framing, and any
    /// body bytes already buffered past the head; the caller streams the
    /// body and decides whether to retry the head.
    async fn write_and_read_head(
        guard: &mut ConnState<U::Conn>,
        header_mode: ResponseHeaderMode,
    ) -> Result<
        (
            u16,
            Vec<(Bytes, Bytes)>,
            proxima_protocols::http1_codec::h1_body::BodyFraming,
            usize,
        ),
        ProximaError,
    > {
        // disjoint borrows of the connection and the buffers — no per-send
        // clone of the encoded request just to satisfy the borrow checker.
        let ConnState {
            conn,
            read_buf,
            write_buf,
            ..
        } = guard;
        let conn = conn
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("connection slot empty".into()))?;
        conn.write_all(write_buf)
            .await
            .map_err(|err| ProximaError::Upstream(format!("write request: {err}")))?;
        conn.flush()
            .await
            .map_err(|err| ProximaError::Upstream(format!("flush request: {err}")))?;
        read_head(conn, read_buf, header_mode).await
    }

    /// Bytes-in / status-out fast path over the keep-alive connection. Writes
    /// the caller's pre-encoded request verbatim, reads just the status +
    /// framing, drains the body to the keep-alive boundary, and returns the
    /// status — allocating NO `Request`, `Response`, or header collection. This
    /// is the load-generator / liveness-prober path: it composes the same
    /// transport and the same framing decoder as [`SendPipe::call`], minus the
    /// request/response envelope the abstraction otherwise builds per call.
    /// Reuses the pooled connection and reconnects once on a stale keep-alive,
    /// exactly like [`Self::exchange`].
    pub async fn send_raw(&self, request_bytes: &[u8]) -> Result<u16, ProximaError> {
        let mut guard = self.state.lock().await;
        let reused = guard.conn.is_some();
        if guard.conn.is_none() {
            let conn = self
                .upstream
                .connect()
                .await
                .map_err(|err| ProximaError::Upstream(format!("connect: {err}")))?;
            guard.conn = Some(conn);
        }
        let first = Self::write_raw_read_status(&mut guard, request_bytes).await;
        let (status, framing, body_offset) = match first {
            Ok(value) => value,
            Err(_) if reused => {
                guard.conn = None;
                let conn = self
                    .upstream
                    .connect()
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("reconnect: {err}")))?;
                guard.conn = Some(conn);
                Self::write_raw_read_status(&mut guard, request_bytes).await?
            }
            Err(error) => return Err(error),
        };
        let result = {
            let ConnState { conn, read_buf, .. } = &mut *guard;
            match conn.as_mut() {
                Some(conn) => drain_body(conn, framing, &read_buf[body_offset..]).await,
                None => Err(ProximaError::Upstream(
                    "connection slot empty after head".into(),
                )),
            }
        };
        match result {
            Ok(()) => Ok(status),
            Err(err) => {
                guard.conn = None;
                Err(err)
            }
        }
    }

    /// Write pre-encoded request bytes and read just the status + framing +
    /// body offset over the connection in `guard`. The `send_raw` analog of
    /// [`Self::write_and_read_head`], skipping the encode and the header copy.
    async fn write_raw_read_status(
        guard: &mut ConnState<U::Conn>,
        request_bytes: &[u8],
    ) -> Result<(u16, proxima_protocols::http1_codec::h1_body::BodyFraming, usize), ProximaError> {
        let ConnState { conn, read_buf, .. } = guard;
        let conn = conn
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("connection slot empty".into()))?;
        conn.write_all(request_bytes)
            .await
            .map_err(|err| ProximaError::Upstream(format!("write request: {err}")))?;
        conn.flush()
            .await
            .map_err(|err| ProximaError::Upstream(format!("flush request: {err}")))?;
        read_head_status(conn, read_buf).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt as _;
    use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    async fn read_request_head(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut scratch = [0_u8; 1024];
        loop {
            let read = socket.read(&mut scratch).await.expect("read req");
            buffer.extend_from_slice(&scratch[..read]);
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                return buffer;
            }
        }
    }

    #[proxima::test]
    async fn get_roundtrip_content_length_returns_status_and_body() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let head = read_request_head(&mut socket).await;
            assert!(head.starts_with(b"GET / HTTP/1.1\r\n"), "got: {head:?}");
            assert!(
                head.windows(b"host: ".len()).any(|w| w == b"host: "),
                "Host header should be injected"
            );
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .await
                .expect("write resp");
        });

        let upstream = TokioTcpUpstream::new(local);
        let client = H1ClientUpstream::new(upstream, "example.com", "test");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = client.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response
            .collect_body()
            .await
            .expect("collect streamed body");
        assert_eq!(&body[..], b"hello");
    }

    /// The streaming claim, proven mechanically (no sleeps): the server writes
    /// the head + the first half of a content-length body, then BLOCKS on a
    /// channel. `call` must return on the head and the body's first chunk must
    /// arrive while the server still withholds the second half — if the client
    /// buffered the whole body, `call` (or the first `next`) would deadlock
    /// waiting for bytes the server hasn't sent, and the test would hang.
    #[proxima::test]
    async fn body_streams_incrementally_before_the_full_response_is_sent() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");
        let (release, hold) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request_head(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\n\r\nAAAAAAAA")
                .await
                .expect("write head + first half");
            socket.flush().await.expect("flush first half");
            // withhold the second half until the client has consumed the first.
            hold.await.ok();
            socket
                .write_all(b"BBBBBBBB")
                .await
                .expect("write second half");
            socket.flush().await.expect("flush second half");
        });

        let client = H1ClientUpstream::new(TokioTcpUpstream::new(local), "example.com", "test");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        // returns on the head — NOT after the (still-incomplete) body.
        let response = client.call(request).await.expect("call");
        assert_eq!(response.status, 200);

        let mut stream = response.into_chunk_stream();
        let first = stream.next().await.expect("first chunk").expect("chunk ok");
        assert_eq!(
            &first[..],
            b"AAAAAAAA",
            "first half streams before the rest is sent"
        );

        // release the rest; the body completes.
        release.send(()).expect("release");
        let mut rest = Vec::new();
        while let Some(chunk) = stream.next().await {
            rest.extend_from_slice(&chunk.expect("chunk ok"));
        }
        assert_eq!(&rest[..], b"BBBBBBBB", "second half streams after release");
    }

    #[proxima::test]
    async fn get_roundtrip_chunked_decodes_body() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request_head(&mut socket).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
                )
                .await
                .expect("write resp");
        });

        let upstream = TokioTcpUpstream::new(local);
        let client = H1ClientUpstream::new(upstream, "example.com", "test");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = client.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response
            .collect_body()
            .await
            .expect("collect streamed body");
        assert_eq!(&body[..], b"hello world");
    }

    /// P14 parity: the streamed body must reproduce, byte-for-byte, what the
    /// old buffered `read_body` returned — across a body larger than one
    /// read chunk (exercises the multi-iteration pump + the keep-alive
    /// hand-back on clean EOF). Real-shaped payload: a JSON array of repeated
    /// records, content-length framed, 40 KiB > READ_CHUNK_BYTES (16 KiB).
    #[proxima::test]
    async fn streamed_body_matches_buffered_across_multiple_reads() {
        let record = b"{\"role\":\"assistant\",\"content\":\"the quick brown fox jumps\"},";
        let mut payload = Vec::with_capacity(40 * 1024);
        payload.extend_from_slice(b"[");
        while payload.len() < 40 * 1024 {
            payload.extend_from_slice(record);
        }
        payload.extend_from_slice(b"]");
        let expected = payload.clone();

        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request_head(&mut socket).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                payload.len()
            );
            socket.write_all(head.as_bytes()).await.expect("write head");
            socket.write_all(&payload).await.expect("write body");
            socket.flush().await.expect("flush");
        });

        let upstream = TokioTcpUpstream::new(local);
        let client = H1ClientUpstream::new(upstream, "example.com", "test");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = client.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response
            .collect_body()
            .await
            .expect("collect streamed body");
        assert_eq!(body.len(), expected.len(), "streamed length must match");
        assert_eq!(
            &body[..],
            &expected[..],
            "streamed bytes must match buffered"
        );
    }

    /// P14 parity: a tiny per-request `timeout` against a server that
    /// accepts but never responds must return `ProximaError::Timeout`,
    /// not hang. Mirrors the hyper `HttpUpstream` timeout race.
    #[proxima::test]
    async fn timeout_against_silent_server_returns_timeout_error() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        // accept the connection, read the request, then sit silent — the
        // client must give up on its own deadline.
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request_head(&mut socket).await;
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(socket);
        });

        let upstream = TokioTcpUpstream::new(local);
        let config = HttpUpstreamConfig {
            timeout: Some(std::time::Duration::from_millis(50)),
            ..HttpUpstreamConfig::default()
        };
        let client = H1ClientUpstream::new(upstream, "example.com", "test").with_config(config);
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let outcome = client.call(request).await;
        assert!(
            matches!(outcome, Err(ProximaError::Timeout(_))),
            "expected timeout, got: {outcome:?}"
        );
    }

    /// P14 parity: an injected request header (template-expanded) must
    /// reach the origin, and the method override must rewrite the request
    /// line. Mirrors the hyper `HttpUpstream` header/method application.
    #[proxima::test]
    async fn injected_header_and_method_override_reach_server() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        let head_slot: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let head_for_server = head_slot.clone();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let head = read_request_head(&mut socket).await;
            *head_for_server.lock().expect("head lock") = head;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .await
                .expect("write resp");
        });

        let mut injected = std::collections::BTreeMap::new();
        injected.insert("x-injected".to_string(), "injected-value".to_string());
        let config = HttpUpstreamConfig {
            method_override: Some("POST".to_string()),
            injected_request_headers: injected,
            ..HttpUpstreamConfig::default()
        };
        let upstream = TokioTcpUpstream::new(local);
        let client = H1ClientUpstream::new(upstream, "example.com", "test").with_config(config);
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = client.call(request).await.expect("call");
        assert_eq!(response.status, 200);

        let head = head_slot.lock().expect("head lock").clone();
        assert!(
            head.starts_with(b"POST / HTTP/1.1\r\n"),
            "method override should rewrite request line; got: {head:?}"
        );
        let needle = b"x-injected: injected-value";
        assert!(
            head.windows(needle.len()).any(|window| window == needle),
            "injected header should reach server; got: {head:?}"
        );
    }

    // ── response-handling config tests ────────────────────────────────────────

    /// The `Drain+Framing` composition (load-gen profile): the client reads
    /// every response byte to advance the keep-alive boundary but returns a
    /// `Response` with no body stream and only the framing headers.
    #[proxima::test]
    async fn drain_framing_discards_body_and_skips_non_framing_headers() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request_head(&mut socket).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\ncontent-type: text/plain\r\nserver: test\r\n\r\nhello",
                )
                .await
                .expect("write resp");
        });

        let client = H1ClientUpstream::new(TokioTcpUpstream::new(local), "example.com", "test")
            .with_response(ResponseHandling::Discard);

        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = client.call(request).await.expect("call");

        assert_eq!(response.status, 200);
        // body was drained — stream is None, payload is empty.
        assert!(response.stream.is_none(), "stream must be None after drain");
        assert!(
            response.payload.is_empty(),
            "payload must be empty after drain"
        );
        // content-length is a framing header — it's kept.
        assert!(
            response.metadata.get_str("content-length").is_some(),
            "content-length is a framing header and must be kept"
        );
        // content-type and server are not framing headers — they're dropped.
        assert!(
            response.metadata.get_str("content-type").is_none(),
            "content-type must be dropped in Framing mode"
        );
        assert!(
            response.metadata.get_str("server").is_none(),
            "server must be dropped in Framing mode"
        );
    }

    /// Keep-alive REUSE under Drain (P14): the connection the internal drain
    /// returns to the pool must sit at a clean message boundary so the NEXT
    /// request reuses it. Fires many sequential Drain requests over one
    /// keep-alive connection — every one must succeed. This is the case the
    /// load bench caught failing (~19% errors) that the single-request drain
    /// test above never exercised. A `Collect` control fires the same load.
    #[proxima::test]
    async fn drain_keep_alive_reuse_across_many_requests() {
        for preset in [ResponseHandling::Discard, ResponseHandling::Full] {
            let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                .await
                .expect("bind");
            let local = listener.local_addr().expect("local_addr");

            // one connection, many requests — mirrors the bench_target server.
            tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.expect("accept");
                let mut scratch = [0_u8; 1024];
                loop {
                    let read = match socket.read(&mut scratch).await {
                        Ok(0) | Err(_) => return,
                        Ok(read) => read,
                    };
                    let terminators = scratch[..read]
                        .windows(4)
                        .filter(|window| *window == b"\r\n\r\n")
                        .count();
                    for _ in 0..terminators {
                        if socket
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok")
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            });

            let client = H1ClientUpstream::new(TokioTcpUpstream::new(local), "example.com", "test")
                .with_response(preset);

            for index in 0..50 {
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("request");
                let response = client
                    .call(request)
                    .await
                    .unwrap_or_else(|err| panic!("{preset:?} request {index} failed: {err}"));
                assert_eq!(response.status, 200, "{preset:?} request {index}");
                // Collect leaves the body on the stream; the caller must drain it
                // for the conn to return to the pool (Drain already did so inside
                // call). Without this the Collect control would itself reconnect.
                if response.stream.is_some() {
                    response
                        .collect_body()
                        .await
                        .unwrap_or_else(|err| panic!("{preset:?} body {index} failed: {err}"));
                }
            }
        }
    }

    /// The bytes-in/status-out fast path reuses the keep-alive connection
    /// across many sends and returns the right status each time — parity with
    /// the `Request`/`Response` Drain path, minus the envelope.
    #[proxima::test]
    async fn send_raw_reuses_keep_alive_and_returns_status() {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = listener.local_addr().expect("local_addr");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut scratch = [0_u8; 1024];
            loop {
                let read = match socket.read(&mut scratch).await {
                    Ok(0) | Err(_) => return,
                    Ok(read) => read,
                };
                let terminators = scratch[..read]
                    .windows(4)
                    .filter(|window| *window == b"\r\n\r\n")
                    .count();
                for _ in 0..terminators {
                    if socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok")
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
        });

        let client = H1ClientUpstream::new(TokioTcpUpstream::new(local), "example.com", "test");
        let request = b"GET / HTTP/1.1\r\nHost: example.com\r\nConnection: keep-alive\r\n\r\n";
        for index in 0..50 {
            let status = client
                .send_raw(request)
                .await
                .unwrap_or_else(|err| panic!("send_raw {index} failed: {err}"));
            assert_eq!(status, 200, "send_raw {index}");
        }
    }

    /// Config-path vs builder-path parity test (P4): building an equivalent
    /// client via conflaguration + `from_config` and via the fluent builder
    /// must yield identical resolved `ResponseHandlingConfig` state.
    #[test]
    fn config_and_builder_paths_produce_identical_response_handling() {
        // builder path: granular knobs.
        let via_builder = H1ClientUpstream::new(
            proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream::new(
                "127.0.0.1:1".parse().expect("addr"),
            ),
            "example.com",
            "test",
        )
        .with_response_body(ResponseBodyMode::Drain)
        .with_response_headers(ResponseHeaderMode::Framing);

        // config path: identical composition from `H1ClientConfig`.
        let config = H1ClientConfig::builder()
            .host("example.com".to_string())
            .response(ResponseHandlingConfig {
                body: ResponseBodyMode::Drain,
                headers: ResponseHeaderMode::Framing,
            })
            .build();
        let via_config = H1ClientUpstream::from_config(
            proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream::new(
                "127.0.0.1:1".parse().expect("addr"),
            ),
            &config,
        );

        assert_eq!(
            via_builder.response_config(),
            via_config.response_config(),
            "builder and config paths must resolve to identical ResponseHandlingConfig"
        );
        assert_eq!(
            via_builder.response_config(),
            ResponseHandlingConfig::from_preset(ResponseHandling::Discard),
            "Drain+Framing must equal the Discard preset"
        );
    }

    /// The preset `ResponseHandling::Full` must equal the default
    /// `Collect+All` — i.e. existing behaviour is unchanged.
    #[test]
    fn full_preset_equals_default_response_handling() {
        let full = ResponseHandlingConfig::from_preset(ResponseHandling::Full);
        let default = ResponseHandlingConfig::default();
        assert_eq!(full, default, "Full preset must equal the default config");
        assert_eq!(full.body, ResponseBodyMode::Collect);
        assert_eq!(full.headers, ResponseHeaderMode::All);
    }

    /// `Discard` preset equals `Drain+Framing`.
    #[test]
    fn discard_preset_equals_drain_framing() {
        let discard = ResponseHandlingConfig::from_preset(ResponseHandling::Discard);
        assert_eq!(discard.body, ResponseBodyMode::Drain);
        assert_eq!(discard.headers, ResponseHeaderMode::Framing);
    }

    /// TOML round-trip: a `ResponseHandlingConfig` serializes and deserializes
    /// to the same value, and the TOML keys match the documented surface.
    #[test]
    fn response_handling_config_toml_round_trip() {
        let cfg = ResponseHandlingConfig {
            body: ResponseBodyMode::Drain,
            headers: ResponseHeaderMode::Framing,
        };
        let toml_str = toml::to_string(&cfg).expect("serialize");
        assert!(
            toml_str.contains("drain"),
            "body = drain must appear in TOML: {toml_str}"
        );
        assert!(
            toml_str.contains("framing"),
            "headers = framing must appear in TOML: {toml_str}"
        );
        let back: ResponseHandlingConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back, cfg);
    }
}

/// Prime-transport keep-alive reuse tests. The tokio-transport tests above
/// prove the drain LOGIC; this exercises the SAME drain path over the real
/// prime reactor (`PrimeTcpUpstream` + `prime::os::net`) — the transport the
/// load generator actually uses — to catch any reactor-level reuse divergence
/// the tokio path would mask. Gated on `stream-client` (the prime upstream dep).
#[cfg(all(test, feature = "http1-stream-client"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod prime_transport_tests {
    use super::*;
    use prime::os::core_shard;
    use prime::os::net::TcpListener as PrimeTcpListener;
    use proxima_net::prime::PrimeTcpUpstream;
    use proxima_runtime::CoreId;
    use std::net::SocketAddr;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const KEEP_ALIVE_OK: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok";

    /// Many sequential Drain requests over one prime keep-alive connection.
    /// Every one must succeed — the connection the internal drain returns to
    /// the pool must be reusable on the prime reactor exactly as it is on
    /// tokio. This is the deterministic analog of the load bench that caught
    /// the Drain path erroring under reuse.
    #[test]
    fn drain_keep_alive_reuse_over_prime_transport() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let outcome: Arc<Mutex<Result<(), String>>> = Arc::new(Mutex::new(Ok(())));
        let addr_chan: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
        let done_factory = done.clone();
        let outcome_factory = outcome.clone();
        let addr_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_factory.clone();
                let outcome = outcome_factory.clone();
                let addr_handle = addr_factory.clone();
                Box::pin(async move {
                    let mut listener =
                        PrimeTcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);

                    // keep-alive server: answer one fixed response per request
                    // head terminator until the client closes (mirrors bench_target).
                    let server = async move {
                        let (mut stream, _peer) = listener.accept().await.expect("accept");
                        let mut scratch = [0_u8; 1024];
                        loop {
                            let read = match stream.read(&mut scratch).await {
                                Ok(0) | Err(_) => return,
                                Ok(read) => read,
                            };
                            let terminators = scratch[..read]
                                .windows(4)
                                .filter(|window| *window == b"\r\n\r\n")
                                .count();
                            for _ in 0..terminators {
                                if stream.write_all(KEEP_ALIVE_OK).await.is_err() {
                                    return;
                                }
                                let _ = stream.flush().await;
                            }
                        }
                    };

                    let client = async move {
                        let h1 = H1ClientUpstream::new(
                            PrimeTcpUpstream::new(bound),
                            "example.com",
                            "test",
                        )
                        .with_response(ResponseHandling::Discard);
                        let mut result = Ok(());
                        for index in 0..50 {
                            let request = Request::builder()
                                .method("GET")
                                .path("/")
                                .build()
                                .expect("request");
                            match h1.call(request).await {
                                Ok(response) if response.status == 200 => {}
                                Ok(response) => {
                                    result =
                                        Err(format!("request {index}: status {}", response.status));
                                    break;
                                }
                                Err(err) => {
                                    result = Err(format!("request {index}: {err}"));
                                    break;
                                }
                            }
                        }
                        *outcome.lock().unwrap() = result;
                    };

                    futures::future::join(server, client).await;
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn core::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if addr_chan.lock().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "prime drain keep-alive round-trip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        handle.shutdown_and_join().expect("shutdown");

        let result = outcome.lock().unwrap().clone();
        assert!(
            result.is_ok(),
            "prime drain keep-alive reuse failed: {result:?}"
        );
    }

    /// The load-generator shape: several Drain clients interleaved on ONE prime
    /// core (the bench fans `connections_per_core` workers into a
    /// `FuturesUnordered` on each core), each its own keep-alive connection.
    /// Every request across every connection must succeed — this is the exact
    /// concurrency condition the load bench ran under when it reported errors.
    #[test]
    fn drain_keep_alive_reuse_concurrent_over_prime_transport() {
        const CONNECTIONS: usize = 5;
        const REQUESTS_PER_CONNECTION: usize = 40;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let outcome: Arc<Mutex<Result<(), String>>> = Arc::new(Mutex::new(Ok(())));
        let addr_chan: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
        let done_factory = done.clone();
        let outcome_factory = outcome.clone();
        let addr_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_factory.clone();
                let outcome = outcome_factory.clone();
                let addr_handle = addr_factory.clone();
                Box::pin(async move {
                    let mut listener =
                        PrimeTcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);

                    // server: accept-loop, one fire-and-forget keep-alive handler
                    // per connection, all on this core.
                    core_shard::spawn_on_current_core(Box::pin(async move {
                        loop {
                            let (mut stream, _peer) = match listener.accept().await {
                                Ok(pair) => pair,
                                Err(_) => return,
                            };
                            core_shard::spawn_on_current_core(Box::pin(async move {
                                let mut scratch = [0_u8; 1024];
                                loop {
                                    let read = match stream.read(&mut scratch).await {
                                        Ok(0) | Err(_) => return,
                                        Ok(read) => read,
                                    };
                                    let terminators = scratch[..read]
                                        .windows(4)
                                        .filter(|window| *window == b"\r\n\r\n")
                                        .count();
                                    for _ in 0..terminators {
                                        if stream.write_all(KEEP_ALIVE_OK).await.is_err() {
                                            return;
                                        }
                                        let _ = stream.flush().await;
                                    }
                                }
                            }));
                        }
                    }));

                    let clients = (0..CONNECTIONS).map(|connection| async move {
                        let h1 = H1ClientUpstream::new(
                            PrimeTcpUpstream::new(bound),
                            "example.com",
                            "test",
                        )
                        .with_response(ResponseHandling::Discard);
                        for index in 0..REQUESTS_PER_CONNECTION {
                            let request = Request::builder()
                                .method("GET")
                                .path("/")
                                .build()
                                .expect("request");
                            match h1.call(request).await {
                                Ok(response) if response.status == 200 => {}
                                Ok(response) => {
                                    return Err(format!(
                                        "conn {connection} request {index}: status {}",
                                        response.status
                                    ));
                                }
                                Err(err) => {
                                    return Err(format!(
                                        "conn {connection} request {index}: {err}"
                                    ));
                                }
                            }
                        }
                        Ok(())
                    });
                    let results = futures::future::join_all(clients).await;
                    let combined = results.into_iter().find(Result::is_err).unwrap_or(Ok(()));
                    *outcome.lock().unwrap() = combined;
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn core::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if addr_chan.lock().unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        while !done.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "concurrent prime drain round-trip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        handle.shutdown_and_join().expect("shutdown");

        let result = outcome.lock().unwrap().clone();
        assert!(
            result.is_ok(),
            "concurrent prime drain keep-alive reuse failed: {result:?}"
        );
    }
}
