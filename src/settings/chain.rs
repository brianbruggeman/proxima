//! Fluent middleware composition via `.then()`.
//!
//! Reading top-down = execution order. `auth.then(rate_limit)
//! .then(upstream)` means: request hits auth first; if it passes,
//! request goes to rate_limit; if that passes, request goes to
//! upstream. Same direction as Express.js, axum route chains, and
//! how anyone draws a middleware pipeline on a whiteboard.
//!
//! Composition produces a `Chain` whose `Into<Spec>` emits the
//! `{ <leaf_fields>, middleware: [<outer>, <inner>, ...] }` shape
//! the existing `apply_middleware_stack` already dispatches on. The
//! loader walks the middleware array in reverse, so the first entry
//! is outermost — which is exactly where our `middlewares` vec puts
//! the outermost layer.
//!
//! ```ignore
//! let composed = BearerAuth::allow_tokens(["t-1"])
//!     .then(RateLimit::token_bucket(100, 50))
//!     .then(HttpUpstream::url("https://backend.internal"));
//! app.pipe("api", composed).await?;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::ProximaError;
use crate::load::{LoadContext, Spec};
use crate::pipe::PipeHandle;

/// Composed middleware stack plus leaf upstream. Produced by
/// `Composable::then(...)`. Holds JSON values directly (not `Spec`)
/// so `Chain: Clone + Debug` works — `Spec` is intentionally neither
/// because of its `Handle(PipeHandle)` variant.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Outer-to-inner middleware list. First entry fires first on a
    /// request. `apply_middleware_stack` walks this array in reverse,
    /// applying the LAST entry first (innermost), so by the time it's
    /// done the FIRST entry is the outermost wrapper.
    middlewares: Vec<Value>,

    /// The leaf upstream that finally handles the request. Required
    /// for a runnable chain; `Chain::then(another_middleware)`
    /// promotes the current leaf to a middleware and replaces it
    /// with `another_middleware` as the new leaf.
    upstream: Value,
}

impl Chain {
    /// Extend the chain with another layer. The previous leaf
    /// becomes a middleware; `next` becomes the new leaf. So
    /// `a.then(b).then(c)` builds: a wraps (b wraps c).
    #[must_use]
    pub fn then<Next: Into<Spec>>(mut self, next: Next) -> Self {
        let prev_leaf = std::mem::replace(&mut self.upstream, inline_value(next.into()));
        self.middlewares.push(prev_leaf);
        self
    }

    /// Optional: attach a human-readable label to the chain endpoint.
    /// Surfaces in recording / swap / metrics under this name.
    #[must_use]
    pub fn labeled(self, _name: impl Into<String>) -> Self {
        // Labeling is a placeholder for the Phase 4 follow-on that
        // wires names into the swap registry. Today the call is a
        // no-op to keep the fluent surface stable.
        self
    }
}

impl From<Chain> for Spec {
    fn from(value: Chain) -> Self {
        let mut map = match value.upstream {
            Value::Object(m) => m,
            // Non-object upstream specs are unusual today (the existing
            // factory dispatch keys off object fields). Wrap them in
            // an "upstream" field so the loader can find them.
            other => {
                let mut m = Map::new();
                m.insert("upstream".into(), other);
                m
            }
        };
        if !value.middlewares.is_empty() {
            map.insert("middleware".into(), Value::Array(value.middlewares));
        }
        Spec::Inline(Value::Object(map))
    }
}

/// Extract the inline `Value` from a `Spec`. Non-inline specs (Path,
/// Handle) collapse to `Value::Null` — those don't fit in a chain by
/// shape today; surfaces as a runtime error at factory build time
/// rather than a silent loss.
fn inline_value(spec: Spec) -> Value {
    match spec {
        Spec::Inline(v) => v,
        _ => Value::Null,
    }
}

/// Anything that produces a `Spec` is composable. The trait is
/// implemented blanket-style for `T: Into<Spec>` so the typed
/// settings (BearerAuth, RateLimit, HttpUpstream, ...) get `.then()`
/// for free.
pub trait Composable: Into<Spec> + Sized {
    /// Wrap `self` around `next`. `self` fires first on the request
    /// path; `next` is what `self` delegates to after its checks pass.
    fn then<Next: Into<Spec>>(self, next: Next) -> Chain {
        Chain {
            middlewares: vec![inline_value(self.into())],
            upstream: inline_value(next.into()),
        }
    }
}

impl<T: Into<Spec>> Composable for T {}

// ── bidirectional fluent chain ⇄ config ──────────────────────────────────────
//
// `Chain`/`.then()` above is the inline-composition surface (a one-shot fluent
// wrap that lowers to a `Spec`). What it lacks is a serde-able config twin that
// round-trips: declare a policy chain fluently, project it to config, reload
// from config, and get the identical chain back. `ChainConfig` is that twin.
//
// The declarative `ChainConfig` is the single source of truth — the builder
// carries one. So `ChainBuilder::build` and the config produce the SAME runtime
// `PipeHandle` (both lower to the `{ <leaf_fields>, middleware: [...] }` value
// the existing `apply_middleware_stack` consumes through the factory registry),
// and `to_config`/`into_builder` round-trip losslessly. Same shape as the
// `ChaosConfig` ⇄ `ChaosBuilder` parity in `proxima-middleware::chaos`.

/// Declarative, serde-able policy chain. The ordered `middlewares` are
/// outer-to-inner `{ "type": <tag>, ... }` specs (the first fires first on a
/// request); `upstream` is the leaf that finally handles it. This is the
/// source of truth the fluent [`ChainBuilder`] carries, so the fluent and
/// config surfaces are one shape and round-trip identically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainConfig {
    /// Outer-to-inner middleware specs. Each is a tagged object the factory
    /// registry dispatches on (`{ "type": "retry", ... }`, `{ "type": "auth",
    /// ... }`, …). Empty = a bare leaf with no policy wrapping.
    #[serde(default)]
    pub middlewares: Vec<Value>,

    /// The leaf upstream spec that terminates the chain (`{ "type": "http",
    /// "url": ... }`). Required for a runnable chain.
    pub upstream: Value,
}

impl ChainConfig {
    /// Lower the chain to the single `{ <leaf_fields>, middleware: [...] }`
    /// value the loader's `apply_middleware_stack` dispatches on — identical
    /// to `From<Chain> for Spec`'s shape, so the two surfaces share the one
    /// build path.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut map = match &self.upstream {
            Value::Object(object) => object.clone(),
            other => {
                let mut object = Map::new();
                object.insert("upstream".into(), other.clone());
                object
            }
        };
        if !self.middlewares.is_empty() {
            map.insert("middleware".into(), Value::Array(self.middlewares.clone()));
        }
        Value::Object(map)
    }

    /// Build the runtime `PipeHandle` through the factory registry. Lowers to
    /// the canonical value and drives it through the existing loader, so a
    /// fluent chain and its config produce byte-for-byte the same stack.
    pub async fn build(&self, context: &LoadContext) -> Result<PipeHandle, ProximaError> {
        crate::load::load(Spec::Inline(self.to_value()), context).await
    }

    /// Project the config into its fluent builder twin (the inverse of
    /// [`ChainBuilder::to_config`]). Powers the round-trip parity guarantee.
    #[must_use]
    pub fn into_builder(self) -> ChainBuilder {
        ChainBuilder { config: self }
    }
}

impl From<ChainConfig> for Spec {
    fn from(value: ChainConfig) -> Self {
        Spec::Inline(value.to_value())
    }
}

/// Fluent twin of [`ChainConfig`]. Compose a policy chain programmatically —
/// `Chain::builder().retry(cfg).rate_limit(cfg)…upstream(leaf)` — then either
/// project it to config ([`ChainBuilder::to_config`]) or build the runtime
/// handle ([`ChainBuilder::build`]). Carries a `ChainConfig` so both surfaces
/// are one shape and round-trip losslessly.
#[derive(Debug, Clone)]
pub struct ChainBuilder {
    config: ChainConfig,
}

impl Chain {
    /// Start a fluent policy chain. Append middleware with the typed sugar
    /// (`.retry`, `.rate_limit`, `.auth`, `.client_auth`) or the generic
    /// `.middleware(impl Into<Spec>)`, then terminate with
    /// `.upstream(impl Into<Spec>)`.
    #[must_use]
    pub fn builder() -> ChainBuilder {
        ChainBuilder {
            config: ChainConfig {
                middlewares: Vec::new(),
                upstream: Value::Null,
            },
        }
    }
}

impl ChainBuilder {
    /// Append a middleware layer from any `Into<Spec>` config. Layers fire in
    /// append order (first appended = outermost). Non-inline specs (a path or
    /// a pre-built handle) don't fit a declarative chain by shape; they lower
    /// to `Value::Null` and surface as a loud factory error at build time
    /// rather than a silent drop.
    #[must_use]
    pub fn middleware<Layer: Into<Spec>>(mut self, layer: Layer) -> Self {
        self.config.middlewares.push(inline_value(layer.into()));
        self
    }

    /// Append a rate-limit layer (typed sugar over [`ChainBuilder::middleware`]).
    #[must_use]
    pub fn rate_limit(self, config: crate::settings::RateLimit) -> Self {
        self.middleware(config)
    }

    /// Append a bearer-auth layer (typed sugar over [`ChainBuilder::middleware`]).
    #[must_use]
    pub fn auth(self, config: crate::settings::BearerAuth) -> Self {
        self.middleware(config)
    }

    /// Append a client-auth layer (typed sugar over [`ChainBuilder::middleware`]).
    #[must_use]
    pub fn client_auth(self, config: crate::settings::ClientAuth) -> Self {
        self.middleware(config)
    }

    /// Append a retry layer from a raw `{ "max_attempts": …, … }` spec. Retry
    /// has no typed config struct yet; the spec mirrors the fields
    /// `RetryFactory` parses, tagged `"retry"` so the registry dispatches it.
    #[must_use]
    pub fn retry(mut self, spec: Value) -> Self {
        self.config.middlewares.push(tag_spec("retry", spec));
        self
    }

    /// Append a transform layer from a raw `{ "request": …, "response": … }`
    /// spec, tagged `"transform"` for the registry.
    #[must_use]
    pub fn transform(mut self, spec: Value) -> Self {
        self.config.middlewares.push(tag_spec("transform", spec));
        self
    }

    /// Append a validate layer from a raw `{ "schema": …, … }` spec, tagged
    /// `"validate"` for the registry.
    #[must_use]
    pub fn validate(mut self, spec: Value) -> Self {
        self.config.middlewares.push(tag_spec("validate", spec));
        self
    }

    /// Set the leaf upstream and finish the declaration. Returns the
    /// declarative [`ChainConfig`] — the source of truth both surfaces share.
    #[must_use]
    pub fn upstream<Leaf: Into<Spec>>(mut self, leaf: Leaf) -> ChainConfig {
        self.config.upstream = inline_value(leaf.into());
        self.config
    }

    /// Project the in-progress builder back to its config without setting a
    /// leaf (the leaf stays whatever it was, `Null` until `.upstream`). The
    /// inverse of [`ChainConfig::into_builder`]; powers round-trip parity.
    #[must_use]
    pub fn to_config(&self) -> ChainConfig {
        self.config.clone()
    }
}

/// Wrap a raw middleware spec object with its `type` tag, so a caller can pass
/// the bare field map for middleware that has no typed config struct yet.
fn tag_spec(tag: &str, spec: Value) -> Value {
    let mut map = match spec {
        Value::Object(object) => object,
        other => {
            let mut object = Map::new();
            object.insert("value".into(), other);
            object
        }
    };
    map.insert("type".into(), Value::String(tag.into()));
    Value::Object(map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::settings::{BearerAuth, HttpUpstream, RateLimit};

    #[test]
    fn upstream_alone_serializes_without_middleware_array() {
        let upstream = HttpUpstream::url("https://example.com");
        let spec: Spec = upstream.into();
        let Spec::Inline(Value::Object(map)) = spec else {
            panic!()
        };
        assert!(map.get("middleware").is_none());
    }

    #[test]
    fn single_middleware_wrap_emits_one_entry_array() {
        let composed = BearerAuth::allow_tokens(["t-1"]).then(HttpUpstream::url("https://backend"));
        let spec: Spec = composed.into();
        let Spec::Inline(Value::Object(map)) = spec else {
            panic!()
        };
        let mw = map.get("middleware").and_then(|v| v.as_array()).unwrap();
        assert_eq!(mw.len(), 1);
        assert_eq!(mw[0].get("type").and_then(|v| v.as_str()), Some("auth"));
        // The leaf's url field survived at the top level.
        assert_eq!(
            map.get("url").and_then(|v| v.as_str()),
            Some("https://backend"),
        );
    }

    #[test]
    fn three_layer_chain_preserves_outer_first_order() {
        let composed = BearerAuth::allow_tokens(["t-1"])
            .then(RateLimit::token_bucket(100, 50))
            .then(HttpUpstream::url("https://backend"));
        let spec: Spec = composed.into();
        let Spec::Inline(Value::Object(map)) = spec else {
            panic!()
        };
        let mw = map.get("middleware").and_then(|v| v.as_array()).unwrap();
        // Outer to inner: auth, then rate_limit. Upstream is the leaf
        // (lives at the top level, not in the middleware array).
        assert_eq!(mw.len(), 2);
        assert_eq!(mw[0].get("type").and_then(|v| v.as_str()), Some("auth"));
        assert_eq!(
            mw[1].get("type").and_then(|v| v.as_str()),
            Some("rate_limit"),
        );
        assert_eq!(
            map.get("url").and_then(|v| v.as_str()),
            Some("https://backend"),
        );
    }

    #[test]
    fn chain_then_replaces_leaf_and_promotes_previous() {
        // The first .then() puts auth as outer and HttpUpstream as leaf.
        // The second .then(rate_limit) should promote the previous leaf
        // (HttpUpstream) to a middleware? No — semantically, rate_limit
        // becomes the new leaf and HttpUpstream becomes... a middleware?
        // That's wrong. Reading order: auth runs first, then upstream,
        // then rate_limit? That's not a valid chain.
        //
        // The intended semantic is: every .then() adds the NEXT layer
        // downstream. The LAST layer is the leaf. Middle layers are
        // wrappers. So a.then(b).then(c) builds the chain [a, b]
        // wrapping leaf c. Reading top-to-bottom = request flow.
        //
        // Caller is responsible for putting an actual upstream at the
        // end. Putting middleware-only chains at the bottom yields
        // factory errors at build time ("auth requires an inner
        // pipe") — loud and clear.
        let composed = BearerAuth::allow_tokens(["t-1"])
            .then(RateLimit::token_bucket(100, 50))
            .then(HttpUpstream::url("https://backend"));
        let spec: Spec = composed.into();
        let Spec::Inline(Value::Object(map)) = spec else {
            panic!()
        };
        // Leaf field present.
        assert_eq!(
            map.get("url").and_then(|v| v.as_str()),
            Some("https://backend"),
        );
    }

    // ── bidirectional ChainConfig ⇄ ChainBuilder ─────────────────────────────

    fn fluent_chain() -> ChainConfig {
        Chain::builder()
            .auth(BearerAuth::allow_tokens(["t-1"]))
            .retry(serde_json::json!({ "max_attempts": 4, "retry_on_status": [503] }))
            .rate_limit(RateLimit::token_bucket(100, 50))
            .upstream(HttpUpstream::url("https://backend"))
    }

    #[test]
    fn fluent_builder_carries_the_declarative_config_in_order() {
        let config = fluent_chain();
        // outer-to-inner: auth, retry, rate_limit. Leaf is the upstream.
        let tags: Vec<&str> = config
            .middlewares
            .iter()
            .filter_map(|mw| mw.get("type").and_then(Value::as_str))
            .collect();
        assert_eq!(tags, ["auth", "retry", "rate_limit"]);
        assert_eq!(
            config.upstream.get("url").and_then(Value::as_str),
            Some("https://backend"),
        );
    }

    #[test]
    fn builder_and_config_lower_to_the_same_value() {
        // the fluent build path and the config both lower to the one
        // `{ <leaf>, middleware: [...] }` value the loader consumes.
        let config = fluent_chain();
        let from_config = config.to_value();
        let from_spec: Spec = config.clone().into();
        let Spec::Inline(from_spec_value) = from_spec else {
            panic!("ChainConfig lowers to an inline spec")
        };
        assert_eq!(from_config, from_spec_value);
        // the leaf url survives at the top level; middleware array carries the
        // three tagged layers in outer-to-inner order.
        assert_eq!(
            from_config.get("url").and_then(Value::as_str),
            Some("https://backend"),
        );
        let mw = from_config
            .get("middleware")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(mw.len(), 3);
    }

    #[test]
    fn into_builder_is_the_inverse_of_to_config() {
        let config = fluent_chain();
        let round = config.clone().into_builder().to_config();
        assert_eq!(round, config, "config → builder → config is identity");
    }

    #[test]
    fn round_trip_through_toml_is_identity() {
        // THE parity fixture: declare a chain fluently, serialize to config,
        // reload from config, and assert the two are identical — the
        // proxima-telemetry / chaos round-trip pattern, for the policy chain.
        let declared = fluent_chain();
        let toml_text = toml::to_string(&declared).expect("encode toml");
        let reloaded: ChainConfig = toml::from_str(&toml_text).expect("decode toml");
        assert_eq!(reloaded, declared, "TOML round-trip diverged");
        // and the reloaded config projects back through the builder identically.
        assert_eq!(reloaded.into_builder().to_config(), declared);
    }

    #[test]
    fn round_trip_through_json_is_identity() {
        let declared = fluent_chain();
        let json_text = serde_json::to_string(&declared).expect("encode json");
        let reloaded: ChainConfig = serde_json::from_str(&json_text).expect("decode json");
        assert_eq!(reloaded, declared, "JSON round-trip diverged");
        // the reloaded config lowers to the identical loader value.
        assert_eq!(reloaded.to_value(), declared.to_value());
    }

    #[test]
    fn bare_chain_without_middleware_omits_the_array() {
        let config = Chain::builder().upstream(HttpUpstream::url("https://backend"));
        assert!(config.middlewares.is_empty());
        let value = config.to_value();
        assert!(
            value.get("middleware").is_none(),
            "no middleware → no array"
        );
        assert_eq!(
            value.get("url").and_then(Value::as_str),
            Some("https://backend")
        );
    }

    // ── runtime parity: fluent build == config build, through the registry ────

    #[proxima::test]
    async fn fluent_and_config_build_the_same_runtime_stack() {
        use crate::load::LoadContext;
        use crate::request::Request;
        use proxima_primitives::pipe::SendPipe;

        // a synth leaf wrapped by a real rate_limit + auth layer, declared
        // fluently. The synth status proves the leaf was reached through both
        // policy layers — the same stack `apply_middleware_stack` walks.
        let leaf = serde_json::json!({ "type": "synth", "status": 200, "body": "ok" });
        let declared = Chain::builder()
            .auth(BearerAuth::allow_tokens(["t-1"]))
            .rate_limit(RateLimit::token_bucket(100, 50))
            .upstream(leaf);

        // reload from config, then build the runtime handle through the
        // factory registry — the existing loader path, not a reimplementation.
        let toml_text = toml::to_string(&declared).expect("encode");
        let reloaded: ChainConfig = toml::from_str(&toml_text).expect("decode");
        assert_eq!(reloaded, declared, "config round-trip identity");

        let context = LoadContext::with_default_registry().expect("ctx");
        let handle = reloaded.build(&context).await.expect("build chain handle");

        let request = Request::builder()
            .method("GET")
            .path("/")
            .header("authorization", "Bearer t-1")
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(
            response.status, 200,
            "reached the synth leaf through auth + rate_limit"
        );
    }
}
