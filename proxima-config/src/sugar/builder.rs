//! The fluent half of the sugar.
//!
//! [`desugar`](crate::sugar::desugar) is the config half — it rewrites a sugary spec
//! `Value` (what a TOML/JSON pipe table deserializes to) into canonical
//! primitives. This module is its mirror: a builder seam that ACCUMULATES
//! that same spec `Value` fluently. Because both halves meet on the one spec
//! `Value`, the fluent builder and the config file are provably the same
//! spec (the "one door" — there is no parallel DSL), and the builder/config
//! parity that principle 4 asks for falls out for free.
//!
//! There is no blanket axis trait here any more (`ProtocolSugar` /
//! `TransportSugar` were removed — a blanket `impl<B: SpecBuilder> Trait for
//! B` adapts an open, unbounded set of foreign types invisibly, which the
//! workspace rules forbid). Each concrete spec builder
//! (`proxima::ListenerBuilder` / `proxima::ClientBuilder`) instead gets its
//! own TYPE-SPECIFIC extension traits (`ListenerTransportExt` /
//! `ClientTransportExt`, etc., in the umbrella crate) that call straight
//! through to [`SpecBuilder::set`]/[`SpecBuilder::push`] below.
//!
//! ```
//! use proxima_config::sugar::SpecBuilder;
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
//! let spec = Spec::default().set("http", "http://api.example.com").set("transport", "tls");
//! assert_eq!(spec.0.get("http").and_then(Value::as_str), Some("http://api.example.com"));
//! assert_eq!(spec.0.get("transport").and_then(Value::as_str), Some("tls"));
//! ```

use serde_json::Value;

/// The base seam every fluent spec builder implements: accumulate keys into the
/// one canonical spec `Value`. Concrete builders implement this once and
/// build their own type-specific axis methods (transport / security /
/// protocol) directly on top — see `proxima::{ListenerBuilder, ClientBuilder}`.
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

    #[test]
    fn set_accumulates_into_the_canonical_spec_map() {
        let fluent = TestSpec::default()
            .set("http", "http://api.example.com")
            .set("transport", "tls")
            .set("proxy", "http://127.0.0.1:8080");
        let from_config = json!({
            "http": "http://api.example.com",
            "transport": "tls",
            "proxy": "http://127.0.0.1:8080",
        });
        assert_eq!(Value::Object(fluent.0), from_config);
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
