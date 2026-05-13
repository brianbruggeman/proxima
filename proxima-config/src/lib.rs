//! Plugin-shaped config format registry. Parsers map source text →
//! `serde_json::Value`; the registry dispatches by file extension or
//! by trial-parse sniff when no hint is given.
//!
//! Tier split: the [`ConfigFormatFactory`] trait, [`DynConfigFormatFactory`],
//! and [`JsonConfigFormat`] compile under `no_std + alloc` — `serde_json`'s
//! `alloc` feature needs no OS. Every other format (`toml`, `json5`, YAML via
//! `serde_norway`, `ron`, `quick-xml`) and the [`ConfigFormatRegistry`] itself
//! (built on `arc-swap`, which is std-only outside its experimental
//! thread-local feature) have no alloc-only upstream equivalent, so they are
//! gated behind `#[cfg(feature = "std")]`. A no_std caller composes formats
//! directly (`JsonConfigFormat.parse(raw)`) rather than through the registry.

#![cfg_attr(
    not(any(feature = "std", feature = "schema-std", feature = "store-std")),
    no_std
)]

// lets `#[derive(Schema)]` (which emits `::proxima_config::schema::…`)
// resolve when a type inside this crate's own `schema` module derives it —
// the standard proc-macro self-reference alias.
extern crate self as proxima_config;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::sync::Arc;

// BTreeMap/Vec/ToString are alloc-tier types, but every call site that needs
// them lives inside the std-gated registry + non-JSON formats below (their
// upstream crates — arc-swap, toml, json5, serde_norway, ron, quick-xml —
// have no alloc-only build), so the import itself is scoped to `std` to keep
// the alloc-only build free of unused-import warnings.
#[cfg(feature = "std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use alloc::string::ToString;
#[cfg(feature = "std")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use arc_swap::ArcSwap;
use serde_json::Value;

use proxima_core::ProximaError;

// pure desugar pass for the TOML/JSON pipe spec (folded in from the former
// proxima-sugar satellite crate) — alloc-tier, same as JsonConfigFormat.
#[cfg(feature = "sugar")]
pub mod sugar;

// typed configuration schemas — folded in from the former proxima-schema
// satellite crate. See `schema/mod.rs` for its own tier split.
#[cfg(feature = "schema")]
pub mod schema;

// k8s-style desired-state file store — folded in from the former
// proxima-state-store satellite crate. See `store/mod.rs` for its own tier
// split.
#[cfg(feature = "store")]
pub mod store;

#[cfg(feature = "alloc")]
pub trait ConfigFormatFactory: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn extensions(&self) -> &[&'static str];
    fn parse(&self, raw: &str) -> Result<Value, ProximaError>;

    /// Serialize a json `Value` back into this format's text — the inverse of
    /// [`parse`](ConfigFormatFactory::parse). The round-trip
    /// `parse(serialize(v)) == v` holds for any value the format can represent.
    /// A format that reads but cannot write is a half-primitive: both directions
    /// are required, so this has no default.
    fn serialize(&self, value: &Value) -> Result<String, ProximaError>;
}

#[cfg(feature = "alloc")]
pub type DynConfigFormatFactory = Arc<dyn ConfigFormatFactory>;

/// three views of the registered formats, kept consistent via a single
/// atomic snapshot. `register` does an atomic copy-on-write that updates
/// all three together so a concurrent `parse_sniff` always sees a
/// coherent (factories, extensions, order) view.
#[cfg(feature = "std")]
struct FormatState {
    factories: BTreeMap<String, DynConfigFormatFactory>,
    extensions: BTreeMap<String, String>,
    order: Vec<String>,
}

#[cfg(feature = "std")]
impl FormatState {
    fn empty() -> Self {
        Self {
            factories: BTreeMap::new(),
            extensions: BTreeMap::new(),
            order: Vec::new(),
        }
    }
}

#[cfg(feature = "std")]
pub struct ConfigFormatRegistry {
    state: ArcSwap<FormatState>,
}

#[cfg(feature = "std")]
impl Default for ConfigFormatRegistry {
    fn default() -> Self {
        Self {
            state: ArcSwap::from_pointee(FormatState::empty()),
        }
    }
}

#[cfg(feature = "std")]
impl ConfigFormatRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, factory: DynConfigFormatFactory) -> Result<(), ProximaError> {
        let name = factory.name().to_string();
        let exts: Vec<String> = factory
            .extensions()
            .iter()
            .map(|ext| (*ext).to_string())
            .collect();
        loop {
            let current = self.state.load_full();
            if current.factories.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "config format `{name}` already registered"
                )));
            }
            let mut factories = current.factories.clone();
            let mut extensions = current.extensions.clone();
            let mut order = current.order.clone();
            factories.insert(name.clone(), factory.clone());
            for ext in &exts {
                extensions.insert(ext.clone(), name.clone());
            }
            order.push(name.clone());
            let next = Arc::new(FormatState {
                factories,
                extensions,
                order,
            });
            let prev = self.state.compare_and_swap(&current, next);
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get_by_name(&self, name: &str) -> Result<DynConfigFormatFactory, ProximaError> {
        self.state
            .load_full()
            .factories
            .get(name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no config format `{name}`")))
    }

    pub fn get_by_extension(&self, ext: &str) -> Result<DynConfigFormatFactory, ProximaError> {
        let snapshot = self.state.load_full();
        let name = snapshot.extensions.get(ext).cloned().ok_or_else(|| {
            ProximaError::Registry(format!("no config format for extension `.{ext}`"))
        })?;
        snapshot
            .factories
            .get(&name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no config format `{name}`")))
    }

    pub fn parse_with_hint(&self, raw: &str, hint: Option<&str>) -> Result<Value, ProximaError> {
        if let Some(name) = hint {
            return self.get_by_name(name)?.parse(raw);
        }
        self.parse_sniff(raw)
    }

    pub fn parse_sniff(&self, raw: &str) -> Result<Value, ProximaError> {
        let snapshot = self.state.load_full();
        let mut errors = Vec::with_capacity(snapshot.order.len());
        for name in &snapshot.order {
            let Some(factory) = snapshot.factories.get(name) else {
                continue;
            };
            match factory.parse(raw) {
                Ok(value) => return Ok(value),
                Err(err) => errors.push(format!("{name}: {err}")),
            }
        }
        Err(ProximaError::Config(format!(
            "could not parse content as any registered format: {}",
            errors.join("; ")
        )))
    }

    /// Serialize a value into the named format — the inverse of
    /// [`parse_with_hint`](ConfigFormatRegistry::parse_with_hint). Errors when the
    /// format is not registered.
    pub fn serialize_with(&self, name: &str, value: &Value) -> Result<String, ProximaError> {
        self.get_by_name(name)?.serialize(value)
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.state.load_full().factories.keys().cloned().collect()
    }
}

/// Registers TOML, JSON, JSON5, YAML, RON, XML. Sniff order matches
/// registration order: TOML is dominant in this repo, JSON is strict
/// (fast-rejects), JSON5 follows JSON (only "wins" on json5-only features),
/// YAML is permissive and goes near the end, RON before XML, XML last.
#[cfg(feature = "std")]
pub fn default_config_format_registry() -> Result<ConfigFormatRegistry, ProximaError> {
    let registry = ConfigFormatRegistry::new();
    registry.register(Arc::new(TomlConfigFormat))?;
    registry.register(Arc::new(JsonConfigFormat))?;
    registry.register(Arc::new(Json5ConfigFormat))?;
    registry.register(Arc::new(YamlConfigFormat))?;
    registry.register(Arc::new(RonConfigFormat))?;
    registry.register(Arc::new(XmlConfigFormat))?;
    Ok(registry)
}

#[cfg(feature = "std")]
pub struct TomlConfigFormat;

#[cfg(feature = "std")]
impl ConfigFormatFactory for TomlConfigFormat {
    fn name(&self) -> &str {
        "toml"
    }
    fn extensions(&self) -> &[&'static str] {
        &["toml"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        let parsed: toml::Value =
            toml::from_str(raw).map_err(|err| ProximaError::Config(format!("toml: {err}")))?;
        Ok(toml_to_json(parsed))
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        toml::to_string(value).map_err(|err| ProximaError::Config(format!("toml: {err}")))
    }
}

#[cfg(feature = "alloc")]
pub struct JsonConfigFormat;

#[cfg(feature = "alloc")]
impl ConfigFormatFactory for JsonConfigFormat {
    fn name(&self) -> &str {
        "json"
    }
    fn extensions(&self) -> &[&'static str] {
        &["json"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        serde_json::from_str(raw).map_err(|err| ProximaError::Config(format!("json: {err}")))
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        serde_json::to_string_pretty(value)
            .map_err(|err| ProximaError::Config(format!("json: {err}")))
    }
}

#[cfg(feature = "std")]
pub struct Json5ConfigFormat;

#[cfg(feature = "std")]
impl ConfigFormatFactory for Json5ConfigFormat {
    fn name(&self) -> &str {
        "json5"
    }
    fn extensions(&self) -> &[&'static str] {
        &["json5"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        json5::from_str(raw).map_err(|err| ProximaError::Config(format!("json5: {err}")))
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        json5::to_string(value).map_err(|err| ProximaError::Config(format!("json5: {err}")))
    }
}

#[cfg(feature = "std")]
pub struct YamlConfigFormat;

#[cfg(feature = "std")]
impl ConfigFormatFactory for YamlConfigFormat {
    fn name(&self) -> &str {
        "yaml"
    }
    fn extensions(&self) -> &[&'static str] {
        &["yaml", "yml"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        let value: Value = serde_norway::from_str(raw)
            .map_err(|err| ProximaError::Config(format!("yaml: {err}")))?;
        // reject bare scalars: yaml accepts almost any string as a scalar,
        // which makes sniff catch bare text. require a mapping/sequence so
        // a non-yaml input doesn't silently parse as `String("garbage")`.
        if !value.is_object() && !value.is_array() {
            return Err(ProximaError::Config(
                "yaml: top-level must be a mapping or sequence".into(),
            ));
        }
        Ok(value)
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        serde_norway::to_string(value).map_err(|err| ProximaError::Config(format!("yaml: {err}")))
    }
}

#[cfg(feature = "std")]
pub struct RonConfigFormat;

#[cfg(feature = "std")]
impl ConfigFormatFactory for RonConfigFormat {
    fn name(&self) -> &str {
        "ron"
    }
    fn extensions(&self) -> &[&'static str] {
        &["ron"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        ron::from_str(raw).map_err(|err| ProximaError::Config(format!("ron: {err}")))
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        ron::to_string(value).map_err(|err| ProximaError::Config(format!("ron: {err}")))
    }
}

#[cfg(feature = "std")]
pub struct XmlConfigFormat;

#[cfg(feature = "std")]
impl ConfigFormatFactory for XmlConfigFormat {
    fn name(&self) -> &str {
        "xml"
    }
    fn extensions(&self) -> &[&'static str] {
        &["xml"]
    }
    fn parse(&self, raw: &str) -> Result<Value, ProximaError> {
        let value: Value = quick_xml::de::from_str(raw)
            .map_err(|err| ProximaError::Config(format!("xml: {err}")))?;
        // quick-xml 0.41 salvages a leading text run from non-xml input as a
        // bare scalar, which makes sniff catch garbage. require a structured
        // element so a non-xml input doesn't silently parse as `String("garbage")`.
        if !value.is_object() && !value.is_array() {
            return Err(ProximaError::Config(
                "xml: top-level must be an element".into(),
            ));
        }
        Ok(value)
    }
    fn serialize(&self, value: &Value) -> Result<String, ProximaError> {
        // xml has no anonymous root; `parse` reads a document whose root tag the
        // deserializer discards, so emit a matching `config` root for symmetry.
        quick_xml::se::to_string_with_root("config", value)
            .map_err(|err| ProximaError::Config(format!("xml: {err}")))
    }
}

#[cfg(feature = "std")]
pub(crate) fn toml_to_json(value: toml::Value) -> Value {
    match value {
        toml::Value::String(text) => Value::String(text),
        toml::Value::Integer(number) => Value::Number(number.into()),
        toml::Value::Float(number) => serde_json::Number::from_f64(number)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        toml::Value::Boolean(flag) => Value::Bool(flag),
        toml::Value::Datetime(timestamp) => Value::String(timestamp.to_string()),
        toml::Value::Array(items) => Value::Array(items.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(table) => Value::Object(
            table
                .into_iter()
                .map(|(key, value)| (key, toml_to_json(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn synth_value() -> Value {
        json!({
            "name": "x",
            "synth": {"status": 200, "body": "hi"}
        })
    }

    #[test]
    fn toml_parses_synth_shape() {
        let raw = "name = \"x\"\n[synth]\nstatus = 200\nbody = \"hi\"\n";
        assert_eq!(TomlConfigFormat.parse(raw).expect("toml"), synth_value());
    }

    #[test]
    fn json_parses_synth_shape() {
        let raw = r#"{"name":"x","synth":{"status":200,"body":"hi"}}"#;
        assert_eq!(JsonConfigFormat.parse(raw).expect("json"), synth_value());
    }

    #[test]
    fn json5_parses_with_comments_and_trailing_commas() {
        let raw = r#"{
            // a comment
            "name": "x",
            "synth": {
                "status": 200,
                "body": "hi",
            },
        }"#;
        assert_eq!(Json5ConfigFormat.parse(raw).expect("json5"), synth_value());
    }

    #[test]
    fn yaml_parses_synth_shape() {
        let raw = "name: x\nsynth:\n  status: 200\n  body: hi\n";
        assert_eq!(YamlConfigFormat.parse(raw).expect("yaml"), synth_value());
    }

    #[test]
    fn yaml_rejects_bare_scalar() {
        let outcome = YamlConfigFormat.parse("just a string");
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn ron_parses_map_shape() {
        let raw = r#"{"name": "x", "synth": {"status": 200, "body": "hi"}}"#;
        assert_eq!(RonConfigFormat.parse(raw).expect("ron"), synth_value());
    }

    #[test]
    fn xml_parses_simple_document() {
        let raw = "<root><name>x</name></root>";
        let value = XmlConfigFormat.parse(raw).expect("xml");
        assert!(value.is_object(), "xml parse must yield an object");
    }

    #[test]
    fn registry_dispatches_by_extension() {
        let registry = default_config_format_registry().expect("registry");
        assert_eq!(
            registry.get_by_extension("toml").expect("toml").name(),
            "toml"
        );
        assert_eq!(
            registry.get_by_extension("json").expect("json").name(),
            "json"
        );
        assert_eq!(
            registry.get_by_extension("json5").expect("json5").name(),
            "json5"
        );
        assert_eq!(
            registry.get_by_extension("yml").expect("yml").name(),
            "yaml"
        );
        assert_eq!(
            registry.get_by_extension("yaml").expect("yaml").name(),
            "yaml"
        );
        assert_eq!(registry.get_by_extension("ron").expect("ron").name(), "ron");
        assert_eq!(registry.get_by_extension("xml").expect("xml").name(), "xml");
        assert!(registry.get_by_extension("unknown").is_err());
    }

    #[test]
    fn registry_sniff_picks_first_success() {
        let registry = default_config_format_registry().expect("registry");
        let raw = r#"{"name":"x","synth":{"status":200,"body":"hi"}}"#;
        let value = registry.parse_sniff(raw).expect("sniff");
        assert_eq!(value, synth_value());
    }

    #[test]
    fn registry_sniff_fails_with_aggregate_error() {
        let registry = default_config_format_registry().expect("registry");
        let outcome = registry.parse_sniff("@#$%^&*not-any-format");
        let Err(ProximaError::Config(msg)) = outcome else {
            panic!("expected aggregate config error");
        };
        assert!(msg.contains("could not parse"));
        assert!(msg.contains("toml"));
        assert!(msg.contains("json"));
    }

    #[test]
    fn registry_parse_with_explicit_hint() {
        let registry = default_config_format_registry().expect("registry");
        let raw = "name: x\nsynth:\n  status: 200\n  body: hi\n";
        let value = registry
            .parse_with_hint(raw, Some("yaml"))
            .expect("yaml hint");
        assert_eq!(value, synth_value());
    }

    #[test]
    fn registry_rejects_duplicate_registration() {
        let registry = ConfigFormatRegistry::new();
        registry
            .register(Arc::new(JsonConfigFormat))
            .expect("first register");
        let outcome = registry.register(Arc::new(JsonConfigFormat));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    // the "both ways or it's wrong" gate: every typed format must round-trip a
    // value for value through serialize -> parse. xml is typeless (`200` -> `"200"`)
    // so it is covered separately for document-survival, not value equality.
    #[test]
    fn typed_formats_round_trip_value_for_value() {
        let value = synth_value();
        let factories: [&dyn ConfigFormatFactory; 5] = [
            &JsonConfigFormat,
            &Json5ConfigFormat,
            &YamlConfigFormat,
            &TomlConfigFormat,
            &RonConfigFormat,
        ];
        for factory in factories {
            let text = factory
                .serialize(&value)
                .unwrap_or_else(|err| panic!("{} serialize: {err}", factory.name()));
            let back = factory
                .parse(&text)
                .unwrap_or_else(|err| panic!("{} parse back: {err}", factory.name()));
            assert_eq!(
                back,
                value,
                "{} must round-trip serialize -> parse value-for-value",
                factory.name()
            );
        }
    }

    #[test]
    fn registry_serialize_with_mirrors_parse_with_hint() {
        let registry = default_config_format_registry().expect("registry");
        let value = synth_value();
        let yaml = registry
            .serialize_with("yaml", &value)
            .expect("serialize yaml");
        assert_eq!(
            registry
                .parse_with_hint(&yaml, Some("yaml"))
                .expect("parse yaml"),
            value
        );
    }

    #[test]
    fn xml_serializes_under_a_config_root_and_reparses() {
        let value = synth_value();
        let text = XmlConfigFormat.serialize(&value).expect("xml serialize");
        assert!(text.contains("<config"), "xml emits a config root: {text}");
        assert!(
            XmlConfigFormat
                .parse(&text)
                .expect("xml parse back")
                .is_object(),
            "xml re-parses to an object"
        );
    }
}
