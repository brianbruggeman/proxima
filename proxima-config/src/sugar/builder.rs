//! The fluent half of the sugar.
//!
//! [`desugar`](crate::sugar::desugar) is the config half — it rewrites a sugary spec
//! `Value` (what a TOML/JSON pipe table deserializes to) into canonical
//! primitives. This module is its mirror: a builder seam plus axis traits that
//! ACCUMULATE that same spec `Value` fluently. Because both halves meet on the
//! one spec `Value`, the fluent builder and the config file are provably the
//! same spec (the "one door" — there is no parallel DSL), and the
//! builder/config parity that principle 4 asks for falls out for free.
//!
//! The sugar is explicit, not magic: each axis is a trait you bring into scope
//! with `use`. The method is on the page because you imported it.
//!
//! ```
//! use proxima_config::sugar::{SpecBuilder, ProtocolSugar, TransportSugar};
//! use serde_json::{Map, Value};
//!
//! #[derive(Default)]
//! struct Spec(Map<String, Value>);
//! impl SpecBuilder for Spec {
//!     fn set(mut self, key: &str, value: impl Into<Value>) -> Self {
//!         self.0.insert(key.to_string(), value.into());
//!         self
//!     }
//!     fn push(mut self, key: &str, value: impl Into<Value>) -> Self {
//!         let entry = self.0.entry(key.to_string()).or_insert_with(|| Value::Array(Vec::new()));
//!         if let Value::Array(array) = entry {
//!             array.push(value.into());
//!         }
//!         self
//!     }
//! }
//!
//! let spec = Spec::default().http("http://api.example.com").tls();
//! assert_eq!(spec.0.get("http").and_then(Value::as_str), Some("http://api.example.com"));
//! assert_eq!(spec.0.get("transport").and_then(Value::as_str), Some("tls"));
//! ```

use alloc::string::String;

use serde_json::Value;

/// The base seam every fluent spec builder implements: accumulate keys into the
/// one canonical spec `Value`. The axis traits ([`ProtocolSugar`],
/// [`TransportSugar`]) are blanket default methods over this — implement `set`
/// and `push` once and a type gets every axis whose trait is in scope.
pub trait SpecBuilder: Sized {
    /// Set a top-level spec key (last write wins) — the fluent twin of a
    /// `key = value` line in the spec table.
    #[must_use]
    fn set(self, key: &str, value: impl Into<Value>) -> Self;

    /// Append to the array under `key` (creating it) — the fluent twin of a
    /// `[[key]]` sequence, e.g. stacking `middleware` entries.
    #[must_use]
    fn push(self, key: &str, value: impl Into<Value>) -> Self;
}

/// Protocol axis (`use proxima_config::sugar::ProtocolSugar`): name the app protocol +
/// upstream url. Lowers to the `http`/`grpc` spec key the factory registry
/// resolves. Blanket-impl'd over every [`SpecBuilder`].
pub trait ProtocolSugar: SpecBuilder {
    /// Point at an HTTP upstream base url (the `http` key).
    #[must_use]
    fn http(self, url: impl Into<String>) -> Self {
        self.set("http", url.into())
    }

    /// HTTPS base url — the `http` key with a `https` scheme; TLS falls out of
    /// the scheme, so this is `.http()` with an https url.
    #[must_use]
    fn https(self, url: impl Into<String>) -> Self {
        self.set("http", url.into())
    }

    /// Point at a gRPC upstream base url (the `grpc` key — gRPC over h2).
    #[must_use]
    fn grpc(self, url: impl Into<String>) -> Self {
        self.set("grpc", url.into())
    }
}

impl<B: SpecBuilder> ProtocolSugar for B {}

/// Transport axis (`use proxima_config::sugar::TransportSugar`): pick the wire under the
/// protocol, and egress routing. Lowers to the `transport` / `proxy` spec keys.
/// Blanket-impl'd over every [`SpecBuilder`]. The transport vocabulary mirrors
/// the typed `Transport` enum in the client (auto/tcp/tls/h3).
pub trait TransportSugar: SpecBuilder {
    /// Negotiate the wire from URL scheme + ALPN (the default).
    #[must_use]
    fn auto(self) -> Self {
        self.set("transport", "auto")
    }

    /// Plaintext TCP.
    #[must_use]
    fn tcp(self) -> Self {
        self.set("transport", "tcp")
    }

    /// TLS over TCP (h1/h2 by ALPN).
    #[must_use]
    fn tls(self) -> Self {
        self.set("transport", "tls")
    }

    /// HTTP/3 over QUIC.
    #[must_use]
    fn h3(self) -> Self {
        self.set("transport", "h3")
    }

    /// Route egress through an HTTP proxy (the `proxy` key) — a CONNECT tunnel
    /// before the upstream.
    #[must_use]
    fn proxy(self, url: impl Into<String>) -> Self {
        self.set("proxy", url.into())
    }
}

impl<B: SpecBuilder> TransportSugar for B {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, Value, json};

    #[derive(Default)]
    struct TestSpec(Map<String, Value>);

    impl SpecBuilder for TestSpec {
        fn set(mut self, key: &str, value: impl Into<Value>) -> Self {
            self.0.insert(key.to_string(), value.into());
            self
        }

        fn push(mut self, key: &str, value: impl Into<Value>) -> Self {
            let entry = self
                .0
                .entry(key.to_string())
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Value::Array(array) = entry {
                array.push(value.into());
            }
            self
        }
    }

    // the load-bearing claim: the fluent axes produce the SAME spec a config
    // file deserializes to (principle 4 parity, principle 7 one-door).
    #[test]
    fn fluent_axes_equal_the_config_spec() {
        let fluent = TestSpec::default()
            .http("http://api.example.com")
            .tls()
            .proxy("http://127.0.0.1:8080");
        let from_config = json!({
            "http": "http://api.example.com",
            "transport": "tls",
            "proxy": "http://127.0.0.1:8080",
        });
        assert_eq!(Value::Object(fluent.0), from_config);
    }

    #[test]
    fn grpc_and_transport_axes_compose() {
        let fluent = TestSpec::default().grpc("https://collector:4317").h3();
        assert_eq!(
            fluent.0.get("grpc").and_then(Value::as_str),
            Some("https://collector:4317")
        );
        assert_eq!(
            fluent.0.get("transport").and_then(Value::as_str),
            Some("h3")
        );
    }

    #[test]
    fn push_stacks_array_entries() {
        let fluent = TestSpec::default()
            .push("middleware", json!({"type": "retry"}))
            .push("middleware", json!({"type": "client-auth"}));
        assert_eq!(
            fluent
                .0
                .get("middleware")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
        assert_eq!(fluent.0["middleware"][0]["type"], "retry");
        assert_eq!(fluent.0["middleware"][1]["type"], "client-auth");
    }
}
