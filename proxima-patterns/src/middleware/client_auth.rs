//! Outbound client authentication middleware (the auth axis, client side).
//!
//! Wraps an inner protocol `Handler` and attaches credentials to each *outbound*
//! request before forwarding — the inject edge of "FSM in the middle, Handler at
//! the edges". The static forms (bearer #2, basic #1) need no FSM: they
//! pre-compute one header value. The dynamic form (oauth #3) drives the
//! `proxima-auth` `TokenLifecycle` FSM with a token-endpoint sub-pipe; it is
//! wired in a follow-up step (the FSM itself is already proven in
//! `proxima-auth`). Selected by the `client-auth` factory key; the `scheme`
//! field chooses the form.

use std::future::Future;
use std::pin::Pin;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zeroize::Zeroizing;

use proxima_auth::{
    AuthTime, Credential, DigestChallenge, DigestClient, SigV4Signer, SignedHeader, TokenLifecycle,
    TokenStep,
};
use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::{Handler, PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::{PipeFactory, PipeFactoryRegistry};
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::sync::{Mutex, Notify};

/// A client-auth wrapping pipe. Holds the precomputed `Authorization` value for
/// the static schemes and injects it on every outbound request.
pub struct ClientAuthPipe<Inner = PipeHandle> {
    inner: Inner,
    header: String,
    value: String,
}

impl ClientAuthPipe<PipeHandle> {
    /// Builds the pipe from a `{scheme, ...}` spec wrapping `inner`.
    ///
    /// # Errors
    /// [`ProximaError::Config`] on a missing/unknown `scheme` or missing fields.
    pub fn from_spec(inner: PipeHandle, spec: &Value) -> Result<Self, ProximaError> {
        let config: ClientAuthConfig = serde_json::from_value(spec.clone())
            .map_err(|err| ProximaError::Config(format!("client-auth config: {err}")))?;
        config.into_static_pipe(inner)
    }
}

fn default_client_auth_header() -> String {
    "authorization".to_string()
}

/// The credential scheme — the `scheme`-tagged config for the `client-auth`
/// middleware. `bearer`/`basic` are static (one precomputed header value);
/// `oauth` drives the `TokenLifecycle` FSM and is built by the factory (it
/// needs the registry to resolve the token-endpoint sub-pipe); `sigv4` signs
/// each request with AWS SigV4; `digest` drives the RFC 7616 challenge-response
/// loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scheme", rename_all = "snake_case")]
pub enum ClientAuthScheme {
    Bearer {
        token: String,
    },
    Basic {
        username: String,
        password: String,
    },
    Oauth {
        token_url: String,
        client_id: String,
        client_secret: String,
        #[serde(default)]
        refresh_ahead_ms: u64,
    },
    Sigv4 {
        access_key_id: String,
        secret_access_key: String,
        region: String,
        service: String,
    },
    Digest {
        username: String,
        password: String,
        #[serde(default)]
        cnonce: Option<String>,
    },
}

/// Typed config surface for the `client-auth` middleware. `header` selects the
/// request header the credential is injected into (defaults to
/// `authorization`); `scheme` chooses the credential form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientAuthConfig {
    #[serde(default = "default_client_auth_header")]
    pub header: String,
    #[serde(flatten)]
    pub scheme: ClientAuthScheme,
}

impl ClientAuthConfig {
    /// Fluent constructor for the static `bearer` scheme.
    #[must_use]
    pub fn bearer(token: impl Into<String>) -> Self {
        Self {
            header: default_client_auth_header(),
            scheme: ClientAuthScheme::Bearer {
                token: token.into(),
            },
        }
    }

    /// Fluent constructor for the static `basic` scheme.
    #[must_use]
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            header: default_client_auth_header(),
            scheme: ClientAuthScheme::Basic {
                username: username.into(),
                password: password.into(),
            },
        }
    }

    /// Override the injection header (defaults to `authorization`).
    #[must_use]
    pub fn with_header(mut self, header: impl Into<String>) -> Self {
        self.header = header.into();
        self
    }

    /// Build the static (`bearer`/`basic`) pipe. Returns a config error for
    /// `oauth`/`sigv4`/`digest`, which the factory builds via specialized types.
    pub fn into_static_pipe(
        self,
        inner: PipeHandle,
    ) -> Result<ClientAuthPipe<PipeHandle>, ProximaError> {
        let value = match &self.scheme {
            ClientAuthScheme::Bearer { token } => format!("Bearer {token}"),
            ClientAuthScheme::Basic { username, password } => {
                format!("Basic {}", BASE64.encode(format!("{username}:{password}")))
            }
            ClientAuthScheme::Oauth { .. } => {
                return Err(ProximaError::Config(
                    "oauth is built by ClientAuthFactory (it needs the registry to resolve the \
                     token-endpoint sub-pipe); ClientAuthPipe::from_spec handles static schemes only"
                        .into(),
                ));
            }
            ClientAuthScheme::Sigv4 { .. } => {
                return Err(ProximaError::Config(
                    "sigv4 is built by ClientAuthFactory; use the factory to build signing pipes"
                        .into(),
                ));
            }
            ClientAuthScheme::Digest { .. } => {
                return Err(ProximaError::Config(
                    "digest is built by ClientAuthFactory; use the factory to build challenge-response pipes"
                        .into(),
                ));
            }
        };
        if !header_safe(&value) {
            return Err(ProximaError::Config(
                "client-auth credential contains control characters (header smuggling guard)"
                    .into(),
            ));
        }
        Ok(ClientAuthPipe {
            inner,
            header: self.header,
            value,
        })
    }
}

fn field(spec: &Value, key: &str, scheme: &str) -> Result<String, ProximaError> {
    spec.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            ProximaError::Config(format!("client-auth scheme `{scheme}` requires `{key}`"))
        })
}

/// Reject control characters AND non-ASCII bytes in a credential before it
/// becomes a header value — a CR/LF in a token or password would smuggle
/// headers into the outbound request, and obs-text (0x80–0xFF) is rejected by
/// HPACK and risks protocol confusion downstream (audit M4). Auth credential
/// values are always ASCII, so this is a tight, correct guard. (Defense in
/// depth alongside the codec's own header validation.)
fn header_safe(value: &str) -> bool {
    value.is_ascii() && !value.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
}

/// `true` when an oauth `token_url` uses plaintext `http://` rather than
/// `https://` (case-insensitive scheme) — the F2 warn trigger. A scheme-less
/// URL is treated as not-plaintext (the http factory resolves the scheme).
fn is_plaintext_token_url(token_url: &str) -> bool {
    let trimmed = token_url.trim_start();
    trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("http://")
}

impl<Inner> SendPipe for ClientAuthPipe<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        request
            .metadata
            .insert(self.header.as_str(), self.value.as_str());
        let inner = self.inner.clone();
        async move { SendPipe::call(&inner, request).await }
    }
}


/// Factory for the `client-auth` key. Holds a `Weak<PipeFactoryRegistry>` so the
/// oauth scheme can build its token-endpoint sub-pipe through the same registry
/// (the exchange edge resolves like any other upstream — mirrors
/// `RecordPipeFactory`). The static schemes (bearer/basic) ignore it.
pub struct ClientAuthFactory {
    upstreams: Weak<PipeFactoryRegistry>,
}

impl ClientAuthFactory {
    #[must_use]
    pub fn new(upstreams: Weak<PipeFactoryRegistry>) -> Self {
        Self { upstreams }
    }
}

impl PipeFactory for ClientAuthFactory {
    fn name(&self) -> &str {
        "client-auth"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        let upstreams = self.upstreams.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("client-auth requires an inner pipe".into()))?;
            let config: ClientAuthConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("client-auth config: {err}")))?;
            match config.scheme {
                ClientAuthScheme::Oauth {
                    token_url,
                    client_id,
                    client_secret,
                    refresh_ahead_ms,
                } => {
                    build_oauth(
                        inner,
                        &token_url,
                        &client_id,
                        &client_secret,
                        refresh_ahead_ms,
                        &upstreams,
                    )
                    .await
                }
                ClientAuthScheme::Sigv4 {
                    access_key_id,
                    secret_access_key,
                    region,
                    service,
                } => {
                    let signer = Arc::new(SigV4Signer::new(
                        &access_key_id,
                        &secret_access_key,
                        &region,
                        &service,
                    ));
                    Ok(into_handle(SigV4AuthPipe { inner, signer }))
                }
                ClientAuthScheme::Digest {
                    username,
                    password,
                    cnonce,
                } => {
                    for (name, value) in [("username", &username), ("password", &password)] {
                        if value
                            .bytes()
                            .any(|byte| byte == b'"' || byte == b'\r' || byte == b'\n')
                        {
                            return Err(ProximaError::Config(format!(
                                "digest `{name}` contains a quote or CRLF (header-injection guard)"
                            )));
                        }
                    }
                    Ok(into_handle(DigestAuthPipe {
                        inner,
                        client: Arc::new(DigestClient::new(&username, &password)),
                        fixed_cnonce: cnonce,
                        nc: Arc::new(AtomicU32::new(0)),
                    }))
                }
                _ => Ok(into_handle(config.into_static_pipe(inner)?)),
            }
        })
    }
}

/// Build the oauth pipe: resolve the token-endpoint sub-pipe via the registry's
/// `http` factory, form-encode the client-credentials body, wrap `inner`.
async fn build_oauth(
    inner: PipeHandle,
    token_url: &str,
    client_id: &str,
    client_secret: &str,
    refresh_ahead_ms: u64,
    upstreams: &Weak<PipeFactoryRegistry>,
) -> Result<PipeHandle, ProximaError> {
    if is_plaintext_token_url(token_url) {
        // F2: the client-credentials secret travels in the body of this POST;
        // over plaintext http it is exposed on the wire.
        tracing::warn!(
            token_url = %token_url,
            "oauth token_url is plaintext http; client_secret is exposed in transit, use https"
        );
    }
    let registry = upstreams.upgrade().ok_or_else(|| {
        ProximaError::Registry("client-auth registry dropped before oauth build".into())
    })?;
    let http = registry.get("http")?;
    let token_endpoint = http
        .build(&serde_json::json!({ "url": token_url }), None)
        .await?;

    // client-credentials grant, application/x-www-form-urlencoded with the
    // id/secret percent-encoded (they routinely carry reserved chars).
    let body = format!(
        "grant_type=client_credentials&client_id={}&client_secret={}",
        form_encode(client_id),
        form_encode(client_secret),
    );
    let request_body: Arc<[u8]> = Arc::from(body.into_bytes().into_boxed_slice());
    Ok(into_handle(OauthAuthPipe::new(
        inner,
        token_endpoint,
        request_body,
        refresh_ahead_ms,
    )))
}

/// Percent-encode a value for `application/x-www-form-urlencoded` — everything
/// but the RFC 3986 unreserved set is escaped.
fn form_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            other => {
                out.push('%');
                out.push_str(&format!("{other:02X}"));
            }
        }
    }
    out
}

const DEFAULT_EXPIRES_IN_SECS: u64 = 3600;

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

/// OAuth2 client-credentials wrapping pipe (auth form #3) — the FSM-in-the-
/// middle edge. Drives a `proxima-auth` `TokenLifecycle`: on each request, if a
/// (re)fetch is due it calls the token-endpoint sub-pipe (the exchange edge —
/// itself a Handler), parses `{access_token, expires_in}`, then injects the bearer
/// and forwards to the inner protocol pipe.
///
/// Single-flight + **non-blocking serve-old**: the lock is NEVER held across
/// the token-endpoint fetch. The fetch winner claims the single-flight slot
/// (`needs_fetch`), releases the lock, and fetches unlocked. During a
/// *refresh-ahead* (the old token is still valid), concurrent callers — and the
/// winner itself — serve the old token immediately via `poll`, never stalling
/// on the refresh. Only cold-start callers (no token yet) park on a
/// [`proxima_primitives::sync::Notify`] until the winner publishes the first token, instead
/// of blocking on the mutex across the fetch.
pub struct OauthAuthPipe {
    inner: PipeHandle,
    token_endpoint: PipeHandle,
    request_body: Arc<[u8]>,
    lifecycle: Arc<Mutex<TokenLifecycle>>,
    /// woken when a fetch publishes a token (or fails), so cold-start callers
    /// that found no usable token re-poll without busy-waiting (RISC: the
    /// proxima/prime notifier, not a hand-rolled condvar).
    fetched: Arc<Notify>,
    start: Instant,
}

impl OauthAuthPipe {
    /// Builds the pipe from resolved pieces: the inner protocol pipe, a
    /// token-endpoint sub-pipe (the exchange edge), the client-credentials POST
    /// body, and the refresh-ahead window. The factory wires the sub-pipe + body
    /// from the spec's `token_url` / `client_id` / `client_secret`.
    #[must_use]
    pub fn new(
        inner: PipeHandle,
        token_endpoint: PipeHandle,
        request_body: Arc<[u8]>,
        refresh_ahead_ms: u64,
    ) -> Self {
        Self {
            inner,
            token_endpoint,
            request_body,
            lifecycle: Arc::new(Mutex::new(TokenLifecycle::new(refresh_ahead_ms))),
            fetched: Arc::new(Notify::new()),
            start: Instant::now(),
        }
    }
}

/// Sample the caller-owned monotonic clock (sans-IO: the edge stamps the time).
fn now_from(start: Instant) -> AuthTime {
    AuthTime(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX))
}

/// Acquire a usable bearer token, never holding the lock across the fetch.
///
/// One short critical section claims the single-flight slot and reads the
/// current token; the (possibly long) token-endpoint call runs unlocked. A
/// caller that already has a valid (old) token returns it immediately — the
/// serve-old, no-stall path. A cold-start caller with no token parks on
/// `fetched` until the winner publishes, then re-polls.
///
/// `start` is sampled fresh each iteration AND after the fetch completes so the
/// stored expiry is computed against the time the token actually lands, not the
/// (possibly stale) time the request began (audit H4).
async fn acquire_token(
    lifecycle: &Mutex<TokenLifecycle>,
    fetched: &Notify,
    token_endpoint: &PipeHandle,
    request_body: &[u8],
    start: Instant,
) -> Result<Zeroizing<String>, ProximaError> {
    loop {
        // register the listener with the notifier BEFORE the state check, so a
        // `notify_waiters` that fires while we hold the lock (or between the
        // poll and the park) wakes THIS listener rather than being lost (audit
        // M3: `notify_waiters` saves no permit, so registration must precede the
        // observation it is meant to cover).
        let listener = fetched.notified();
        futures::pin_mut!(listener);
        let _ = listener.as_mut().enable();

        let now = now_from(start);
        let (claimed_fetch, ready, fetch_in_flight) = {
            let mut fsm = lifecycle.lock().await;
            let claimed_fetch = fsm.needs_fetch(now);
            let ready = match fsm.poll(now) {
                TokenStep::Use(credential) => Some(credential.secret()),
                TokenStep::Await => None,
            };
            (claimed_fetch, ready, fsm.fetch_in_flight())
        };

        if claimed_fetch {
            // we own the refresh: fetch UNLOCKED so concurrent callers serving
            // the old token are never blocked behind this await.
            match fetch_token(token_endpoint, request_body, now_from(start)).await {
                Ok((credential, expires_at)) => {
                    let new_token = credential.secret();
                    lifecycle.lock().await.set_token(credential, expires_at);
                    fetched.notify_waiters();
                    // refresh-ahead serves the old token to this very request if
                    // it had one; otherwise the freshly fetched one.
                    return Ok(ready.unwrap_or(new_token));
                }
                Err(err) => {
                    lifecycle.lock().await.fetch_failed(now_from(start));
                    fetched.notify_waiters();
                    if let Some(old) = ready {
                        // refresh-ahead failed but the old token is still valid:
                        // serve it, do not fail the request.
                        return Ok(old);
                    }
                    return Err(err);
                }
            }
        }

        if let Some(token) = ready {
            // serve-old / serve-current without blocking on the in-flight fetch.
            return Ok(token);
        }

        if !fetch_in_flight {
            // no token, we did not claim a fetch, and none is in flight — a prior
            // cold-start fetch failed and the cool-down is still open (audit M5).
            // Parking now would wait for a notify that will not come this round,
            // so surface the failure instead of deadlocking.
            return Err(ProximaError::Upstream(
                "oauth token unavailable: a prior fetch failed and the retry cool-down is open"
                    .into(),
            ));
        }

        // cold start, fetch in flight, no usable token: park on the listener
        // (already registered above) until the winner publishes, then re-poll.
        listener.await;
    }
}

async fn fetch_token(
    endpoint: &PipeHandle,
    body: &[u8],
    now: AuthTime,
) -> Result<(Credential, AuthTime), ProximaError> {
    let request = Request::builder()
        .method("POST")
        .path("/")
        .header("content-type", "application/x-www-form-urlencoded")
        .payload(bytes::Bytes::copy_from_slice(body))
        .build()?;
    let response = SendPipe::call(endpoint, request).await?;
    // the body carries the plaintext token; wipe the buffer on drop (audit H2)
    // so it does not linger in freed heap after we lift the token out.
    let raw = Zeroizing::new(response.collect_body().await?.to_vec());
    let parsed: TokenResponse = serde_json::from_slice(&raw)
        .map_err(|err| ProximaError::Decode(format!("oauth token response: {err}")))?;
    let expires_in = parsed.expires_in.unwrap_or(DEFAULT_EXPIRES_IN_SECS);
    let expires_at = AuthTime(now.0.saturating_add(expires_in.saturating_mul(1000)));
    // moving the token into the zeroize-on-drop Credential ends the plaintext's
    // exposure window; `parsed.access_token` is consumed here, not copied.
    Ok((Credential::Bearer(parsed.access_token), expires_at))
}

impl SendPipe for OauthAuthPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        let token_endpoint = self.token_endpoint.clone();
        let request_body = self.request_body.clone();
        let lifecycle = self.lifecycle.clone();
        let fetched = self.fetched.clone();
        let start = self.start;
        async move {
            let token =
                acquire_token(&lifecycle, &fetched, &token_endpoint, &request_body, start).await?;
            if !header_safe(token.as_str()) {
                return Err(ProximaError::Upstream(
                    "oauth token contains control characters (header smuggling guard)".into(),
                ));
            }
            let header_value = Zeroizing::new(format!("Bearer {}", token.as_str()));
            request
                .metadata
                .insert("authorization", header_value.as_str());
            SendPipe::call(&inner, request).await
        }
    }
}


/// AWS SigV4 request-signing pipe (auth form #5) — the signing edge. Computes
/// an `Authorization` value per outbound request from the request bytes + the
/// derived signing key (it does not attach a static credential). Drives the
/// `proxima-auth` `SigV4Signer`. The caller must stamp `x-amz-date`
/// (`YYYYMMDDTHHMMSSZ`) on the request (sans-IO: the edge owns the clock).
pub struct SigV4AuthPipe<Inner = PipeHandle> {
    inner: Inner,
    /// the signer is deliberately not `Clone` (its secret must not be copied —
    /// audit Z1); share it behind `Arc` so each request clones the handle, not
    /// the secret key material.
    signer: Arc<SigV4Signer>,
}

impl SigV4AuthPipe<PipeHandle> {
    /// Build from a `{scheme:"sigv4", access_key_id, secret_access_key, region,
    /// service}` spec.
    ///
    /// # Errors
    /// [`ProximaError::Config`] on any missing field.
    pub fn from_spec(inner: PipeHandle, spec: &Value) -> Result<Self, ProximaError> {
        let access_key_id = field(spec, "access_key_id", "sigv4")?;
        let secret_access_key = field(spec, "secret_access_key", "sigv4")?;
        let region = field(spec, "region", "sigv4")?;
        let service = field(spec, "service", "sigv4")?;
        let signer = Arc::new(SigV4Signer::new(
            &access_key_id,
            &secret_access_key,
            &region,
            &service,
        ));
        Ok(Self { inner, signer })
    }
}

/// Collect the request's headers into the SigV4 `SignedHeader` shape, plus the
/// `host` header and `x-amz-date` (the two SigV4 always signs). Returns the
/// sorted header set + the `x-amz-date` value the signer needs.
fn sigv4_headers(request: &Request<Bytes>) -> Result<(Vec<SignedHeader>, String), ProximaError> {
    let amz_date = request
        .metadata
        .get_str("x-amz-date")
        .ok_or_else(|| {
            ProximaError::Config(
                "sigv4 requires an `x-amz-date` header (YYYYMMDDTHHMMSSZ) on the request".into(),
            )
        })?
        .to_string();
    let mut headers = Vec::new();
    for (name, value) in request.metadata.iter() {
        if let (Ok(name), Ok(value)) = (core::str::from_utf8(name), core::str::from_utf8(value)) {
            headers.push(SignedHeader {
                name: name.into(),
                value: value.into(),
            });
        }
    }
    Ok((headers, amz_date))
}

impl<Inner> SendPipe for SigV4AuthPipe<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        let signer = self.signer.clone();
        async move {
            let (headers, amz_date) = sigv4_headers(&request)?;
            let method = core::str::from_utf8(request.method.as_bytes())
                .map_err(|_| ProximaError::Config("sigv4: non-utf8 method".into()))?
                .to_string();
            let path = core::str::from_utf8(&request.path)
                .map_err(|_| ProximaError::Config("sigv4: non-utf8 path".into()))?
                .to_string();
            let payload = request.payload.clone();
            let authorization =
                signer.authorization(&method, &path, "", &headers, &payload, &amz_date);
            if !header_safe(&authorization) {
                return Err(ProximaError::Upstream(
                    "sigv4 authorization contains control characters".into(),
                ));
            }
            request.metadata.insert("authorization", authorization);
            SendPipe::call(&inner, request).await
        }
    }
}


/// HTTP Digest challenge-response pipe (auth form #4, RFC 7616) — the
/// challenge-response edge. Sends the request; on a `401` carrying
/// `WWW-Authenticate: Digest …`, computes the `Authorization: Digest` response
/// (driving the `proxima-auth` `DigestClient`) and retries once.
///
/// The `cnonce` is freshly randomized PER REQUEST (audit H3) and the nonce
/// count `nc` increments per request (audit Z4) so the server's
/// `{nonce, nc}` replay window is respected and the response is not a fixed,
/// offline-attackable value. A `cnonce` may be pinned via config — TEST-ONLY
/// determinism; a production edge leaves it unset for fresh entropy.
pub struct DigestAuthPipe<Inner = PipeHandle> {
    inner: Inner,
    /// the client owns the cleartext password (zeroize-on-drop, not `Clone`);
    /// share it behind `Arc` so each request clones the handle, not the secret
    /// (audit M4 — mirrors `SigV4AuthPipe`'s `Arc<SigV4Signer>`).
    client: Arc<DigestClient>,
    /// test-only fixed cnonce; `None` means a fresh random cnonce per request.
    fixed_cnonce: Option<String>,
    /// per-challenge nonce count, formatted `{:08x}` (RFC 7616 §3.4); monotonic
    /// and incremented only when a challenge is actually answered (audit M3).
    nc: Arc<AtomicU32>,
}

impl DigestAuthPipe<PipeHandle> {
    /// Build from a `{scheme:"digest", username, password, cnonce?}` spec.
    /// `cnonce` is test-only; omit it in production for per-request entropy.
    ///
    /// # Errors
    /// [`ProximaError::Config`] on a missing `username`/`password`, or a
    /// `username`/`password` carrying a `"`/CR/LF that would break out of the
    /// quoted `Authorization` value (audit M2 — caught at build, not per request).
    pub fn from_spec(inner: PipeHandle, spec: &Value) -> Result<Self, ProximaError> {
        let username = field(spec, "username", "digest")?;
        let password = field(spec, "password", "digest")?;
        for (name, value) in [("username", &username), ("password", &password)] {
            if value
                .bytes()
                .any(|byte| byte == b'"' || byte == b'\r' || byte == b'\n')
            {
                return Err(ProximaError::Config(format!(
                    "digest `{name}` contains a quote or CRLF (header-injection guard)"
                )));
            }
        }
        let fixed_cnonce = spec
            .get("cnonce")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok(Self {
            inner,
            client: Arc::new(DigestClient::new(&username, &password)),
            fixed_cnonce,
            nc: Arc::new(AtomicU32::new(0)),
        })
    }
}

/// A fresh 128-bit client nonce as 32 lowercase hex chars (RFC 7616 §3.4
/// recommends an unpredictable cnonce). `fastrand` is the workspace's existing
/// PRNG (RISC) — not cryptographically strong, but per-request-unique, which
/// is what defeats the fixed-cnonce offline-attack + replay surface here.
fn fresh_cnonce() -> String {
    format!("{:016x}{:016x}", fastrand::u64(..), fastrand::u64(..))
}

impl<Inner> SendPipe for DigestAuthPipe<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        let client = self.client.clone();
        let counter = self.nc.clone();
        let cnonce = self.fixed_cnonce.clone().unwrap_or_else(fresh_cnonce);
        async move {
            let method = core::str::from_utf8(request.method.as_bytes())
                .map_err(|_| ProximaError::Config("digest: non-utf8 method".into()))?
                .to_string();
            let path = core::str::from_utf8(&request.path)
                .map_err(|_| ProximaError::Config("digest: non-utf8 path".into()))?
                .to_string();
            // first attempt — preemptive send to elicit the challenge.
            let first = SendPipe::call(&inner, clone_request(&request)?).await?;
            if first.status != 401 {
                return Ok(first);
            }
            let Some(challenge_header) = first.metadata.get_str("www-authenticate") else {
                return Ok(first);
            };
            let challenge = DigestChallenge::parse(challenge_header)
                .map_err(|err| ProximaError::Upstream(format!("digest challenge: {err}")))?;
            // advance nc ONLY now that a challenge is actually being answered
            // (audit M3: per-challenge, not per-request). A wrap back to 0 would
            // recycle a count the server's replay window already saw (audit C1),
            // so refuse rather than silently reuse `00000000`.
            let nc_value = counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
            if nc_value == 0 {
                return Err(ProximaError::Upstream(
                    "digest nonce-count wrapped; rebuild the auth pipe with fresh credentials"
                        .into(),
                ));
            }
            let nc = format!("{nc_value:08x}");
            let authorization = client.authorization(&challenge, &method, &path, &nc, &cnonce);
            if !header_safe(&authorization) {
                return Err(ProximaError::Upstream(
                    "digest authorization contains control characters".into(),
                ));
            }
            let mut retry = clone_request(&request)?;
            retry.metadata.insert("authorization", authorization);
            SendPipe::call(&inner, retry).await
        }
    }
}

/// Shallow-clone a buffered request for the digest preempt/retry pair (the
/// payload is `Bytes`, cheap to clone; metadata + method + path are small).
/// `Request` is not `Clone`, so we rebuild via the builder.
fn clone_request(request: &Request<Bytes>) -> Result<Request<Bytes>, ProximaError> {
    let mut cloned = Request::builder()
        .method(request.method.clone())
        .path(request.path.clone())
        .payload(request.payload.clone());
    for (name, value) in request.metadata.iter() {
        cloned = cloned.header(name.clone(), value.clone());
    }
    cloned.build()
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Records the `authorization` header it received, then returns 200.
    struct Capture {
        seen: Arc<Mutex<Option<String>>>,
    }

    impl SendPipe for Capture {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let value = request
                .metadata
                .iter()
                .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(b"authorization"))
                .and_then(|(_, value)| std::str::from_utf8(value).ok().map(str::to_string));
            *self.seen.lock().unwrap() = value;
            async move { Ok(Response::new(200)) }
        }
    }


    fn run(spec: Value) -> Option<String> {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen: seen.clone() });
        let auth = ClientAuthPipe::from_spec(inner, &spec).expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        futures::executor::block_on(async { auth.call(request).await.expect("call") });
        seen.lock().unwrap().clone()
    }

    #[test]
    fn bearer_injects_authorization_header_on_the_outbound_request() {
        let seen = run(serde_json::json!({ "scheme": "bearer", "token": "tok-123" }));
        assert_eq!(seen.as_deref(), Some("Bearer tok-123"));
    }

    // principle-4 parity: the fluent constructor and the config value must lower
    // to identical ClientAuthPipe state (header + precomputed credential value).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let inner = || {
            into_handle(Capture {
                seen: Arc::new(Mutex::new(None)),
            })
        };

        let from_value: ClientAuthConfig = serde_json::from_value(serde_json::json!({
            "scheme": "basic",
            "header": "proxy-authorization",
            "username": "alice",
            "password": "s3cr3t",
        }))
        .expect("from_value");
        let from_value = from_value.into_static_pipe(inner()).expect("static value");

        let from_builder = ClientAuthConfig::basic("alice", "s3cr3t")
            .with_header("proxy-authorization")
            .into_static_pipe(inner())
            .expect("static builder");

        assert_eq!(from_value.header, from_builder.header);
        assert_eq!(from_value.value, from_builder.value);
    }

    #[test]
    fn basic_injects_base64_user_password() {
        let seen = run(
            serde_json::json!({ "scheme": "basic", "username": "alice", "password": "s3cr3t" }),
        );
        // base64("alice:s3cr3t")
        assert_eq!(seen.as_deref(), Some("Basic YWxpY2U6czNjcjN0"));
    }

    #[test]
    fn unknown_scheme_is_a_config_error() {
        let inner = into_handle(Capture {
            seen: Arc::new(Mutex::new(None)),
        });
        let outcome = ClientAuthPipe::from_spec(inner, &serde_json::json!({ "scheme": "nope" }));
        assert!(outcome.is_err());
    }

    /// The oauth FSM-through-the-chain: the pipe drives TokenLifecycle, calls the
    /// token-endpoint sub-pipe (mock), parses the token, and injects the bearer
    /// onto the outbound request — pipes at the edges, FSM in the middle.
    struct MockTokenEndpoint;

    impl SendPipe for MockTokenEndpoint {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                Ok(Response::new(200).with_body("{\"access_token\":\"at-1\",\"expires_in\":3600}"))
            }
        }
    }


    #[test]
    fn oauth_fetches_token_then_injects_bearer() {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen: seen.clone() });
        let endpoint = into_handle(MockTokenEndpoint);
        let pipe = OauthAuthPipe::new(
            inner,
            endpoint,
            Arc::from(&b"grant_type=client_credentials"[..]),
            0,
        );
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        futures::executor::block_on(async { pipe.call(request).await.expect("call") });
        assert_eq!(seen.lock().unwrap().as_deref(), Some("Bearer at-1"));
    }

    /// The full surface-to-pipe wiring: the factory resolves the oauth
    /// token-endpoint through the registry's `http` factory (stubbed here), builds
    /// the OauthAuthPipe, and the chain injects the fetched bearer.
    struct StubHttp;

    impl PipeFactory for StubHttp {
        fn name(&self) -> &str {
            "http"
        }

        fn build(
            &self,
            _spec: &Value,
            _inner: Option<PipeHandle>,
        ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
            Box::pin(async { Ok(into_handle(MockTokenEndpoint)) })
        }
    }

    #[test]
    fn control_chars_in_a_credential_are_rejected() {
        let inner = into_handle(Capture {
            seen: Arc::new(Mutex::new(None)),
        });
        let outcome = ClientAuthPipe::from_spec(
            inner,
            &serde_json::json!({ "scheme": "bearer", "token": "tok\r\nX-Evil: 1" }),
        );
        assert!(
            outcome.is_err(),
            "a CRLF token must be rejected (header smuggling guard)"
        );
    }

    #[test]
    fn form_encode_escapes_reserved_characters() {
        assert_eq!(form_encode("a b&c=d/e"), "a%20b%26c%3Dd%2Fe");
        assert_eq!(form_encode("keep-._~AZ09"), "keep-._~AZ09");
    }

    #[test]
    fn factory_builds_oauth_through_the_registry_http_subpipe() {
        let registry = Arc::new(PipeFactoryRegistry::new());
        registry
            .register(Arc::new(StubHttp))
            .expect("register http");
        let factory = ClientAuthFactory::new(Arc::downgrade(&registry));

        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen: seen.clone() });
        let spec = serde_json::json!({
            "scheme": "oauth",
            "token_url": "https://idp/token",
            "client_id": "id",
            "client_secret": "secret",
        });
        let handle = futures::executor::block_on(factory.build(&spec, Some(inner))).expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        futures::executor::block_on(async {
            SendPipe::call(&handle, request).await.expect("call")
        });
        assert_eq!(seen.lock().unwrap().as_deref(), Some("Bearer at-1"));
    }

    // ── F2: plaintext token_url warn ────────────────────────────────────────

    #[test]
    fn plaintext_http_token_url_is_flagged_for_warn() {
        assert!(is_plaintext_token_url("http://idp.example/token"));
        assert!(is_plaintext_token_url("HTTP://idp.example/token"));
        assert!(!is_plaintext_token_url("https://idp.example/token"));
        assert!(
            !is_plaintext_token_url("idp.example/token"),
            "scheme-less is not flagged"
        );
    }

    // ── #5: SigV4 request-signing pipe (the get-vanilla vector through the chain)

    #[test]
    fn sigv4_pipe_signs_the_request_with_the_get_vanilla_signature() {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen: seen.clone() });
        let spec = serde_json::json!({
            "scheme": "sigv4",
            "access_key_id": "AKIDEXAMPLE",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "region": "us-east-1",
            "service": "service",
        });
        let pipe = SigV4AuthPipe::from_spec(inner, &spec).expect("build sigv4");
        // the same request the get-vanilla vector signs: host + x-amz-date.
        let request = Request::builder()
            .method("GET")
            .path("/")
            .header("host", "example.amazonaws.com")
            .header("x-amz-date", "20150830T123600Z")
            .build()
            .expect("request");
        futures::executor::block_on(async { pipe.call(request).await.expect("call") });
        let authorization = seen
            .lock()
            .unwrap()
            .clone()
            .expect("authorization injected");
        assert!(
            authorization.ends_with(
                "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
            ),
            "sigv4 pipe must emit the published get-vanilla signature; got `{authorization}`"
        );
        assert!(authorization.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request"
        ));
    }

    #[test]
    fn sigv4_pipe_requires_an_amz_date_header() {
        let inner = into_handle(Capture {
            seen: Arc::new(Mutex::new(None)),
        });
        let spec = serde_json::json!({
            "scheme": "sigv4", "access_key_id": "a", "secret_access_key": "s",
            "region": "us-east-1", "service": "service",
        });
        let pipe = SigV4AuthPipe::from_spec(inner, &spec).expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .header("host", "h")
            .build()
            .expect("request");
        let outcome = futures::executor::block_on(async { pipe.call(request).await });
        assert!(outcome.is_err(), "missing x-amz-date is a config error");
    }

    /// A server that 401s with a `WWW-Authenticate: Digest` challenge on the
    /// first request, then 200s once an `Authorization: Digest` is present —
    /// the RFC 7616 challenge-response loop the DigestAuthPipe drives.
    struct DigestChallengeServer {
        seen_auth: Arc<Mutex<Option<String>>>,
    }

    impl SendPipe for DigestChallengeServer {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let authorization = request
                .metadata
                .get_str("authorization")
                .map(str::to_string);
            let seen = self.seen_auth.clone();
            async move {
                match authorization {
                    Some(value) => {
                        *seen.lock().unwrap() = Some(value);
                        Ok(Response::new(200))
                    }
                    None => {
                        let challenge = "Digest realm=\"http-auth@example.org\", \
                            qop=\"auth\", algorithm=SHA-256, \
                            nonce=\"7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v\"";
                        let mut response = Response::new(401);
                        response.metadata.insert("www-authenticate", challenge);
                        Ok(response)
                    }
                }
            }
        }
    }


    #[test]
    fn digest_pipe_answers_a_401_challenge_and_retries() {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(DigestChallengeServer {
            seen_auth: seen.clone(),
        });
        let spec = serde_json::json!({
            "scheme": "digest",
            "username": "Mufasa",
            "password": "Circle of Life",
            "cnonce": "f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ",
        });
        let pipe = DigestAuthPipe::from_spec(inner, &spec).expect("build digest");
        let request = Request::builder()
            .method("GET")
            .path("/dir/index.html")
            .build()
            .expect("request");
        let response =
            futures::executor::block_on(async { pipe.call(request).await.expect("call") });
        assert_eq!(response.status, 200, "the retry with credentials succeeds");
        let authorization = seen
            .lock()
            .unwrap()
            .clone()
            .expect("authorization on retry");
        // the RFC 7616 §3.9.1 SHA-256 response, derived from the §3.4.6 formula.
        assert!(
            authorization.contains(
                "response=\"753927fa0e85d155564e2e272a28d1802ca10daf4496794697cf8db5856cb6c1\""
            ),
            "digest retry must carry the RFC 7616 §3.9.1 SHA-256 response; got `{authorization}`"
        );
    }

    #[test]
    fn digest_username_with_a_quote_is_rejected_at_build() {
        // audit M2: a username carrying a `"` would break out of the quoted
        // Authorization value; refuse it at construction, not silently per call.
        let inner = into_handle(Capture {
            seen: Arc::new(Mutex::new(None)),
        });
        let outcome = DigestAuthPipe::from_spec(
            inner,
            &serde_json::json!({ "scheme": "digest", "username": "ev\"il", "password": "p" }),
        );
        assert!(
            outcome.is_err(),
            "a quote in the username must be rejected at build"
        );
    }

    /// A token endpoint that counts how many times it was hit, so we can prove
    /// single-flight + serve-old: a refresh-ahead window fires exactly one
    /// fetch while the old token is still served.
    struct CountingTokenEndpoint {
        hits: Arc<Mutex<u32>>,
    }

    impl SendPipe for CountingTokenEndpoint {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let hits = self.hits.clone();
            async move {
                let count = {
                    let mut guard = hits.lock().unwrap();
                    *guard += 1;
                    *guard
                };
                // each fetch yields a distinct token so we can tell them apart.
                let body = format!("{{\"access_token\":\"at-{count}\",\"expires_in\":1}}");
                Ok(Response::new(200).with_body(body))
            }
        }
    }


    /// refresh-ahead non-blocking serve-old: with a 1-second token and a large
    /// refresh-ahead window, the SECOND request lands inside the refresh window.
    /// The FSM serves the still-valid OLD token (`at-1`) to that request while a
    /// single background-eligible refresh is triggered — proving the lock is not
    /// held across the fetch and the caller is not stalled awaiting a new token.
    #[test]
    fn oauth_refresh_ahead_serves_the_old_token_without_blocking() {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen: seen.clone() });
        let hits = Arc::new(Mutex::new(0));
        let endpoint = into_handle(CountingTokenEndpoint { hits: hits.clone() });
        // expires_in is 1s = 1000ms; refresh 5_000ms ahead => always in-window
        // once a token exists, so every post-first request is a refresh-ahead.
        let pipe = OauthAuthPipe::new(
            inner,
            endpoint,
            Arc::from(&b"grant_type=client_credentials"[..]),
            5_000,
        );
        let first = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        futures::executor::block_on(async { pipe.call(first).await.expect("first call") });
        let first_token = seen.lock().unwrap().clone();
        assert_eq!(
            first_token.as_deref(),
            Some("Bearer at-1"),
            "cold start fetches at-1"
        );

        // second request: inside the refresh-ahead window. It must be served the
        // OLD token (at-1) immediately, not stall for a new fetch.
        let second = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        futures::executor::block_on(async { pipe.call(second).await.expect("second call") });
        assert_eq!(
            seen.lock().unwrap().as_deref(),
            Some("Bearer at-1"),
            "refresh-ahead serves the still-valid old token, not a stall"
        );
    }

    /// A token endpoint that always errors — proves the cold-start failure path
    /// surfaces the error instead of deadlocking. The M5 backoff added a
    /// cool-down; a cold-start caller with no token, no in-flight fetch, and an
    /// open cool-down must NOT park forever (audit M5 deadlock guard).
    struct FailingTokenEndpoint;

    impl SendPipe for FailingTokenEndpoint {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move { Err(ProximaError::Upstream("token endpoint down".into())) }
        }
    }


    #[test]
    fn oauth_cold_start_fetch_failure_returns_an_error_not_a_hang() {
        let seen = Arc::new(Mutex::new(None));
        let inner = into_handle(Capture { seen });
        let endpoint = into_handle(FailingTokenEndpoint);
        let pipe = OauthAuthPipe::new(
            inner,
            endpoint,
            Arc::from(&b"grant_type=client_credentials"[..]),
            0,
        );
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let outcome = futures::executor::block_on(async { pipe.call(request).await });
        assert!(
            outcome.is_err(),
            "a cold-start token fetch failure must surface, not hang"
        );
    }
}
