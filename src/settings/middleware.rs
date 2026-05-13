//! Typed middleware configs. Each is a `bon::Builder`-derived struct
//! whose `Into<Spec>` impl produces the `{ "type": "<tag>", ... }`
//! shape the existing factory registry already dispatches on. Same
//! pattern as `HttpUpstream`; middleware factories take an inner
//! Pipe to wrap, so the standalone build path uses `App::pipe`
//! with the chain-composition pattern (Phase 4 wires `.then()` for
//! inline composition).
//!
//! Today: `BearerAuth`, `RateLimit`. Others (Retry, Transform,
//! Isolate, Validate, Diff) follow the same pattern.

use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::load::Spec;

/// Bearer-token authentication middleware. Wraps an inner Pipe;
/// rejects requests whose `Authorization` header does not match one
/// of the allowed tokens.
///
/// ```ignore
/// let auth = BearerAuth::builder().allow(vec!["t-1".into(), "t-2".into()]).build();
/// ```
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct BearerAuth {
    /// Token allow-list. At least one required by the factory.
    pub allow: Vec<String>,

    /// Header name. Default `"authorization"` (case-insensitive
    /// match).
    #[serde(default)]
    pub header: Option<String>,

    /// `WWW-Authenticate` realm string. Default `"proxima"`.
    #[serde(default)]
    pub realm: Option<String>,

    /// Status code for rejected requests. Default `401`.
    #[serde(default)]
    pub on_unauthorized_status: Option<u16>,

    /// Prefix to strip from the header value before comparing. Default
    /// `"Bearer "`. Set to empty string to disable.
    #[serde(default)]
    pub strip_prefix: Option<String>,
}

impl BearerAuth {
    /// Shorthand for the common case: just an allow-list.
    pub fn allow_tokens<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::builder()
            .allow(tokens.into_iter().map(Into::into).collect())
            .build()
    }
}

impl From<BearerAuth> for Spec {
    fn from(value: BearerAuth) -> Self {
        let mut map = Map::new();
        map.insert("type".into(), Value::String("auth".into()));
        map.insert(
            "allow".into(),
            Value::Array(value.allow.into_iter().map(Value::String).collect()),
        );
        if let Some(header) = value.header {
            map.insert("header".into(), Value::String(header));
        }
        if let Some(realm) = value.realm {
            map.insert("realm".into(), Value::String(realm));
        }
        if let Some(status) = value.on_unauthorized_status {
            map.insert("on_unauthorized_status".into(), Value::from(status));
        }
        if let Some(strip) = value.strip_prefix {
            map.insert("strip_prefix".into(), Value::String(strip));
        }
        Spec::Inline(Value::Object(map))
    }
}

/// Token-bucket rate-limit middleware. `capacity` is the burst size,
/// `refill_per_sec` is the steady-state rate. `key` selects which
/// request attribute partitions the rate-limit space.
///
/// ```ignore
/// let rl = RateLimit::builder().capacity(100).refill_per_sec(50).build();
/// ```
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct RateLimit {
    /// Bucket capacity (burst).
    pub capacity: u64,

    /// Steady-state refill rate in tokens / sec.
    pub refill_per_sec: u64,

    /// Key extractor — partitions the bucket space. One of:
    /// `"path_and_method"` (default), `"constant"`, `"header"`.
    #[serde(default)]
    pub key: Option<String>,

    /// Required if `key = "constant"`. Single bucket for all traffic
    /// under this constant string.
    #[serde(default)]
    pub constant_value: Option<String>,

    /// Required if `key = "header"`. Per-header-value bucket.
    #[serde(default)]
    pub header_name: Option<String>,

    /// `Retry-After` header value in ms on rejected requests.
    #[serde(default)]
    pub retry_after_ms: Option<u64>,
}

impl RateLimit {
    /// Shorthand: token bucket with `capacity` burst and
    /// `refill_per_sec` steady-state rate, keyed by path + method.
    #[must_use]
    pub fn token_bucket(capacity: u64, refill_per_sec: u64) -> Self {
        Self::builder()
            .capacity(capacity)
            .refill_per_sec(refill_per_sec)
            .build()
    }
}

impl From<RateLimit> for Spec {
    fn from(value: RateLimit) -> Self {
        let mut map = Map::new();
        map.insert("type".into(), Value::String("rate_limit".into()));
        map.insert("capacity".into(), Value::from(value.capacity));
        map.insert("refill_per_sec".into(), Value::from(value.refill_per_sec));
        let key = value.key.unwrap_or_else(|| "path_and_method".to_string());
        map.insert("key".into(), Value::String(key));
        if let Some(constant) = value.constant_value {
            map.insert("constant_value".into(), Value::String(constant));
        }
        if let Some(header) = value.header_name {
            map.insert("header_name".into(), Value::String(header));
        }
        if let Some(ms) = value.retry_after_ms {
            map.insert("retry_after_ms".into(), Value::from(ms));
        }
        Spec::Inline(Value::Object(map))
    }
}

/// OAuth2 client-credentials parameters (auth form #3) — its own `bon` builder,
/// so the fluent surface is real builder sugar:
/// `OauthAuth::builder().token_url(..).client_id(..).client_secret(..).build()`.
/// The exchange endpoint is itself a Pipe; the live token is refreshed before
/// expiry via the `proxima-auth` `TokenLifecycle` FSM.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct OauthAuth {
    /// token endpoint the exchange edge calls (client-credentials grant)
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
    /// refresh this many ms before expiry so the live token covers the refetch
    /// latency. Default 0 (refresh on expiry).
    #[serde(default)]
    #[builder(default)]
    pub refresh_ahead_ms: u64,
}

/// Outbound client authentication — the auth axis. First-class on both surfaces
/// (principle 4): serde/TOML config AND fluent builder sugar
/// (`OauthAuth::builder()…` for the field-heavy form, `bearer`/`basic`
/// one-liners for the rest), with a round-trip identity. Lowers to a
/// `client-auth` wrapping pipe that drives a `proxima-auth` FSM underneath.
///
/// ```ignore
/// // fluent builder sugar
/// client.auth(ClientAuth::bearer("tok"));
/// client.auth(OauthAuth::builder().token_url("https://idp/token")
///     .client_id("id").client_secret("secret").refresh_ahead_ms(30_000).build());
/// // config (TOML)
/// // [[middleware]]
/// // type = "client-auth"
/// // scheme = "oauth"
/// // token_url = "https://idp/token"
/// ```
///
/// `ClientAuth` is the flat config-mirror (TOML-clean: a `scheme` tag + the
/// per-form fields). `OauthAuth` lowers into it via `From`, so the oauth fluent
/// path is the typed builder while config stays a plain table.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct ClientAuth {
    /// `"bearer"` · `"basic"` · `"oauth"` · `"sigv4"` · `"digest"`.
    pub scheme: String,
    /// bearer: the token value.
    #[serde(default)]
    pub token: Option<String>,
    /// basic / digest: username.
    #[serde(default)]
    pub username: Option<String>,
    /// basic / digest: password.
    #[serde(default)]
    pub password: Option<String>,
    /// oauth: token endpoint (the exchange edge — a nested Pipe).
    #[serde(default)]
    pub token_url: Option<String>,
    /// oauth: client id.
    #[serde(default)]
    pub client_id: Option<String>,
    /// oauth: client secret.
    #[serde(default)]
    pub client_secret: Option<String>,
    /// oauth: refresh-ahead window in ms (default 0).
    #[serde(default)]
    pub refresh_ahead_ms: Option<u64>,
    /// sigv4: AWS access-key id (public, travels in the credential scope).
    #[serde(default)]
    pub access_key_id: Option<String>,
    /// sigv4: AWS secret access key.
    #[serde(default)]
    pub secret_access_key: Option<String>,
    /// sigv4: AWS region (e.g. `us-east-1`).
    #[serde(default)]
    pub region: Option<String>,
    /// sigv4: AWS service code (e.g. `s3`).
    #[serde(default)]
    pub service: Option<String>,
    /// digest: client nonce (the edge rotates it; config pins it for replay).
    #[serde(default)]
    pub cnonce: Option<String>,
}

impl ClientAuth {
    /// Static bearer token — auth form #2.
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::builder().scheme("bearer").token(token).build()
    }

    /// HTTP Basic — auth form #1 (request-level).
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self::builder()
            .scheme("basic")
            .username(username)
            .password(password)
            .build()
    }
}

impl From<OauthAuth> for ClientAuth {
    /// The oauth fluent-builder result lowers into the flat config-mirror.
    fn from(value: OauthAuth) -> Self {
        Self::builder()
            .scheme("oauth")
            .token_url(value.token_url)
            .client_id(value.client_id)
            .client_secret(value.client_secret)
            .refresh_ahead_ms(value.refresh_ahead_ms)
            .build()
    }
}

impl From<SigV4Auth> for ClientAuth {
    /// The sigv4 fluent-builder result lowers into the flat config-mirror.
    fn from(value: SigV4Auth) -> Self {
        Self::builder()
            .scheme("sigv4")
            .access_key_id(value.access_key_id)
            .secret_access_key(value.secret_access_key)
            .region(value.region)
            .service(value.service)
            .build()
    }
}

impl From<DigestAuth> for ClientAuth {
    /// The digest fluent-builder result lowers into the flat config-mirror.
    fn from(value: DigestAuth) -> Self {
        Self::builder()
            .scheme("digest")
            .username(value.username)
            .password(value.password)
            .maybe_cnonce(value.cnonce)
            .build()
    }
}

impl From<ClientAuth> for Spec {
    fn from(value: ClientAuth) -> Self {
        let mut map = Map::new();
        map.insert("type".into(), Value::String("client-auth".into()));
        map.insert("scheme".into(), Value::String(value.scheme));
        for (key, field) in [
            ("token", value.token),
            ("username", value.username),
            ("password", value.password),
            ("token_url", value.token_url),
            ("client_id", value.client_id),
            ("client_secret", value.client_secret),
            ("access_key_id", value.access_key_id),
            ("secret_access_key", value.secret_access_key),
            ("region", value.region),
            ("service", value.service),
            ("cnonce", value.cnonce),
        ] {
            if let Some(text) = field {
                map.insert(key.into(), Value::String(text));
            }
        }
        if let Some(ms) = value.refresh_ahead_ms {
            map.insert("refresh_ahead_ms".into(), Value::from(ms));
        }
        Spec::Inline(Value::Object(map))
    }
}

/// AWS SigV4 request-signing parameters (auth form #5) — its own `bon` builder.
/// The secret access key signs each request; the credential is computed per
/// request, not static. Lowers into [`ClientAuth`] via `From`.
///
/// ```ignore
/// client.auth(SigV4Auth::builder()
///     .access_key_id("AKID…").secret_access_key("…")
///     .region("us-east-1").service("s3").build());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct SigV4Auth {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
    pub service: String,
}

/// HTTP Digest (RFC 7616) parameters (auth form #4) — its own `bon` builder.
/// The challenge-response pipe answers a `401 WWW-Authenticate: Digest`. Lowers
/// into [`ClientAuth`] via `From`.
///
/// ```ignore
/// client.auth(DigestAuth::builder().username("u").password("p").build());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct DigestAuth {
    pub username: String,
    pub password: String,
    /// client nonce; the edge rotates it per exchange, config pins it.
    #[serde(default)]
    pub cnonce: Option<String>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bearer_auth_serializes_with_type_tag() {
        let bearer = BearerAuth::allow_tokens(["t-1", "t-2"]);
        let spec: Spec = bearer.into();
        let Spec::Inline(value) = spec else {
            panic!("expected inline")
        };
        assert_eq!(value.get("type").and_then(|v| v.as_str()), Some("auth"));
        let allow = value
            .get("allow")
            .and_then(|v| v.as_array())
            .expect("allow array");
        assert_eq!(allow.len(), 2);
    }

    #[test]
    fn bearer_auth_round_trips_through_toml() {
        let original = BearerAuth::builder()
            .allow(vec!["a".into(), "b".into()])
            .header("x-token")
            .realm("acme")
            .on_unauthorized_status(403)
            .strip_prefix("Token ")
            .build();
        let toml_text = toml::to_string(&original).expect("encode");
        let restored: BearerAuth = toml::from_str(&toml_text).expect("decode");
        assert_eq!(restored.allow, original.allow);
        assert_eq!(restored.header, original.header);
        assert_eq!(restored.realm, original.realm);
        assert_eq!(
            restored.on_unauthorized_status,
            original.on_unauthorized_status
        );
        assert_eq!(restored.strip_prefix, original.strip_prefix);
    }

    #[test]
    fn rate_limit_serializes_with_type_tag() {
        let rl = RateLimit::token_bucket(100, 50);
        let spec: Spec = rl.into();
        let Spec::Inline(value) = spec else {
            panic!("expected inline")
        };
        assert_eq!(
            value.get("type").and_then(|v| v.as_str()),
            Some("rate_limit"),
        );
        assert_eq!(value.get("capacity").and_then(|v| v.as_u64()), Some(100));
        assert_eq!(
            value.get("refill_per_sec").and_then(|v| v.as_u64()),
            Some(50),
        );
    }

    #[test]
    fn rate_limit_round_trips_through_toml() {
        let original = RateLimit::builder()
            .capacity(500)
            .refill_per_sec(100)
            .key("header")
            .header_name("x-client-id")
            .retry_after_ms(2_000)
            .build();
        let toml_text = toml::to_string(&original).expect("encode");
        let restored: RateLimit = toml::from_str(&toml_text).expect("decode");
        assert_eq!(restored.capacity, original.capacity);
        assert_eq!(restored.refill_per_sec, original.refill_per_sec);
        assert_eq!(restored.key, original.key);
        assert_eq!(restored.header_name, original.header_name);
        assert_eq!(restored.retry_after_ms, original.retry_after_ms);
    }

    #[test]
    fn oauth_builder_sugar_lowers_into_flat_client_auth() {
        let auth: ClientAuth = OauthAuth::builder()
            .token_url("https://idp/token")
            .client_id("id")
            .client_secret("secret")
            .refresh_ahead_ms(30_000)
            .build()
            .into();
        assert_eq!(auth.scheme, "oauth");
        assert_eq!(auth.token_url.as_deref(), Some("https://idp/token"));
        assert_eq!(auth.client_id.as_deref(), Some("id"));
        assert_eq!(auth.refresh_ahead_ms, Some(30_000));
    }

    #[test]
    fn client_auth_serializes_with_type_and_scheme_tags() {
        let spec: Spec = ClientAuth::bearer("tok").into();
        let Spec::Inline(value) = spec else {
            panic!("expected inline")
        };
        assert_eq!(
            value.get("type").and_then(|v| v.as_str()),
            Some("client-auth")
        );
        assert_eq!(value.get("scheme").and_then(|v| v.as_str()), Some("bearer"));
        assert_eq!(value.get("token").and_then(|v| v.as_str()), Some("tok"));
    }

    #[test]
    fn client_auth_round_trips_through_toml_every_scheme() {
        // the principle-4 invariant: config ⇄ value identity across all forms.
        let cases = [
            ClientAuth::bearer("tok"),
            ClientAuth::basic("alice", "s3cr3t"),
            OauthAuth::builder()
                .token_url("https://idp/token")
                .client_id("id")
                .client_secret("secret")
                .refresh_ahead_ms(5_000)
                .build()
                .into(),
            SigV4Auth::builder()
                .access_key_id("AKIDEXAMPLE")
                .secret_access_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
                .region("us-east-1")
                .service("s3")
                .build()
                .into(),
            DigestAuth::builder()
                .username("Mufasa")
                .password("Circle of Life")
                .build()
                .into(),
        ];
        for original in cases {
            let toml_text = toml::to_string(&original).expect("encode");
            let restored: ClientAuth = toml::from_str(&toml_text).expect("decode");
            assert_eq!(restored, original, "config round-trip identity");
        }
    }

    #[test]
    fn sigv4_builder_sugar_lowers_into_flat_client_auth() {
        let auth: ClientAuth = SigV4Auth::builder()
            .access_key_id("AKIDEXAMPLE")
            .secret_access_key("secret")
            .region("us-east-1")
            .service("s3")
            .build()
            .into();
        assert_eq!(auth.scheme, "sigv4");
        assert_eq!(auth.access_key_id.as_deref(), Some("AKIDEXAMPLE"));
        assert_eq!(auth.region.as_deref(), Some("us-east-1"));
        assert_eq!(auth.service.as_deref(), Some("s3"));
    }

    #[test]
    fn digest_builder_sugar_lowers_into_flat_client_auth() {
        let auth: ClientAuth = DigestAuth::builder()
            .username("Mufasa")
            .password("Circle of Life")
            .build()
            .into();
        assert_eq!(auth.scheme, "digest");
        assert_eq!(auth.username.as_deref(), Some("Mufasa"));
        assert_eq!(auth.password.as_deref(), Some("Circle of Life"));
    }

    /// Principle 4: the fluent builder surface and the config (Spec) surface
    /// produce the identical spec for sigv4 — the both-ways parity fixture.
    #[test]
    fn sigv4_fluent_and_config_surfaces_agree() {
        let fluent: Spec = ClientAuth::from(
            SigV4Auth::builder()
                .access_key_id("AKID")
                .secret_access_key("sk")
                .region("us-east-1")
                .service("s3")
                .build(),
        )
        .into();
        let Spec::Inline(fluent_value) = fluent else {
            panic!("expected inline")
        };
        assert_eq!(
            fluent_value.get("type").and_then(Value::as_str),
            Some("client-auth")
        );
        assert_eq!(
            fluent_value.get("scheme").and_then(Value::as_str),
            Some("sigv4")
        );
        assert_eq!(
            fluent_value.get("region").and_then(Value::as_str),
            Some("us-east-1")
        );
        assert_eq!(
            fluent_value.get("service").and_then(Value::as_str),
            Some("s3")
        );
    }
}
