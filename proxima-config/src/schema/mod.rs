//! Schema primitive — value-shape IR in serde's data model. Used by the
//! `validate` middleware to gate request bodies and by `proxima describe`
//! to emit equivalent contracts in JSON Schema / OpenAPI / TOML / etc.
//!
//! Folded in from the former `proxima-schema` satellite crate. Feature-gated
//! independently of the format-registry tiers above: `schema` is the alloc
//! floor (IR, `Describe`, emit), `schema-std` adds `SchemaRegistry` / the toml
//! emitter / the `serde_value` escape hatch, `schema-derive` adds
//! `#[derive(Schema)]`.

#[cfg(feature = "schema")]
pub mod describe;
#[cfg(feature = "schema")]
pub mod emit;
#[cfg(feature = "schema-std")]
pub mod scenario;

#[cfg(feature = "schema")]
pub use describe::{Describe, field};
#[cfg(feature = "schema-std")]
pub use scenario::register_scenario_schemas;

/// `#[derive(Schema)]` — generate a [`Schema`] for a type (the derive lives in
/// the macro namespace, the IR enum in the type namespace, so they share the
/// name the way serde's `Serialize` trait and derive do). emits an
/// `impl Describe`, so bring both into scope: `use proxima_config::schema::{Schema, Describe};`.
/// enabled by the `schema-derive` feature.
#[cfg(feature = "schema-derive")]
pub use proxima_macros::Schema;

#[cfg(feature = "schema")]
use alloc::boxed::Box;
#[cfg(feature = "schema-std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "schema")]
use alloc::format;
#[cfg(feature = "schema")]
use alloc::string::{String, ToString};
#[cfg(feature = "schema-std")]
use alloc::sync::Arc;
#[cfg(feature = "schema")]
use alloc::vec::Vec;

#[cfg(feature = "schema-std")]
use arc_swap::ArcSwap;
#[cfg(feature = "schema")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "schema")]
use serde_json::Value;

#[cfg(feature = "schema-std")]
use proxima_core::ProximaError;

#[cfg(feature = "schema")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Schema {
    Bool,
    Int {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<i64>,
    },
    UInt {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u64>,
    },
    Float {
        #[serde(default)]
        finite: bool,
    },
    String {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<StringFormat>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_len: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_len: Option<usize>,
    },
    Bytes {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_len: Option<usize>,
    },
    #[serde(alias = "array")]
    Seq {
        items: Box<Schema>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_items: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_items: Option<usize>,
    },
    Map {
        keys: Box<Schema>,
        values: Box<Schema>,
    },
    Tuple(Vec<Schema>),
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
    },
    Optional(Box<Schema>),
    Ref(String),
    Any,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StringFormat {
    Email,
    Url,
    Uuid,
    Ipv4,
    Ipv6,
    Date,
    DateTime,
    Hostname,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub schema: Schema,
    #[serde(default)]
    pub flags: FieldFlags,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldFlags {
    /// when true the field may be absent from an instance value
    #[serde(default)]
    pub optional: bool,
    /// surface in emitted schemas as deprecated; not enforced at validate
    #[serde(default)]
    pub deprecated: bool,
    /// human-readable description, surfaced by emitters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: String,
    /// when None, the variant is a unit (matched by string equality)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Box<Schema>>,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub path: Vec<PathSegment>,
    pub message: String,
}

#[cfg(feature = "schema")]
#[derive(Debug, Clone, PartialEq)]
pub enum PathSegment {
    Field(String),
    Index(usize),
}

#[cfg(feature = "schema")]
impl ValidationError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            path: Vec::new(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn at_field(mut self, name: impl Into<String>) -> Self {
        self.path.insert(0, PathSegment::Field(name.into()));
        self
    }

    #[must_use]
    pub fn at_index(mut self, index: usize) -> Self {
        self.path.insert(0, PathSegment::Index(index));
        self
    }

    #[must_use]
    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            return "$".into();
        }
        let mut buffer = String::from("$");
        for segment in &self.path {
            match segment {
                PathSegment::Field(name) => {
                    buffer.push('.');
                    buffer.push_str(name);
                }
                PathSegment::Index(index) => {
                    buffer.push('[');
                    buffer.push_str(&index.to_string());
                    buffer.push(']');
                }
            }
        }
        buffer
    }
}

#[cfg(feature = "schema")]
impl core::fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}: {}", self.path_string(), self.message)
    }
}

#[cfg(feature = "schema")]
impl core::error::Error for ValidationError {}

/// Resolves `Ref` schemas during validation.
#[cfg(feature = "schema")]
pub trait SchemaResolver {
    fn resolve(&self, name: &str) -> Option<Schema>;
}

/// A resolver with no registered shapes — `Ref` resolution always fails.
#[cfg(feature = "schema")]
pub struct EmptyResolver;

#[cfg(feature = "schema")]
impl SchemaResolver for EmptyResolver {
    fn resolve(&self, _name: &str) -> Option<Schema> {
        None
    }
}

#[cfg(feature = "schema")]
impl Schema {
    /// Validate any serializable value against this schema. generic over the
    /// value type, so callers pass a typed struct, a `serde_value::Value`, or a
    /// config-loaded value — anything `Serialize` — never a json-coupled type.
    /// the value is normalized once into the serde data model (the canonical
    /// shape the validator walks); that intermediate is a transient, not a
    /// public type. `resolver` resolves any `Ref` shapes during traversal.
    pub fn validate<T>(
        &self,
        value: &T,
        resolver: &dyn SchemaResolver,
    ) -> Result<(), ValidationError>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(value).map_err(|err| {
            ValidationError::new(format!("value not serializable for validation: {err}"))
        })?;
        let mut visiting: Vec<String> = Vec::new();
        self.validate_inner(&value, resolver, &mut visiting)
    }

    fn validate_inner(
        &self,
        value: &Value,
        resolver: &dyn SchemaResolver,
        visiting: &mut Vec<String>,
    ) -> Result<(), ValidationError> {
        match self {
            Self::Bool => match value {
                Value::Bool(_) => Ok(()),
                _ => Err(ValidationError::new("expected bool")),
            },
            Self::Int { min, max } => match value.as_i64() {
                Some(number) => {
                    if let Some(lower) = *min
                        && number < lower
                    {
                        return Err(ValidationError::new(format!("must be >= {lower}")));
                    }
                    if let Some(upper) = *max
                        && number > upper
                    {
                        return Err(ValidationError::new(format!("must be <= {upper}")));
                    }
                    Ok(())
                }
                None => Err(ValidationError::new("expected signed integer")),
            },
            Self::UInt { min, max } => match value.as_u64() {
                Some(number) => {
                    if let Some(lower) = *min
                        && number < lower
                    {
                        return Err(ValidationError::new(format!("must be >= {lower}")));
                    }
                    if let Some(upper) = *max
                        && number > upper
                    {
                        return Err(ValidationError::new(format!("must be <= {upper}")));
                    }
                    Ok(())
                }
                None => Err(ValidationError::new("expected unsigned integer")),
            },
            Self::Float { finite } => match value.as_f64() {
                Some(number) => {
                    if *finite && !number.is_finite() {
                        return Err(ValidationError::new("must be a finite number"));
                    }
                    Ok(())
                }
                None => Err(ValidationError::new("expected float")),
            },
            Self::String {
                pattern,
                format,
                min_len,
                max_len,
            } => {
                let text = value
                    .as_str()
                    .ok_or_else(|| ValidationError::new("expected string"))?;
                let chars = text.chars().count();
                if let Some(lower) = *min_len
                    && chars < lower
                {
                    return Err(ValidationError::new(format!(
                        "string length {chars} < min_len {lower}"
                    )));
                }
                if let Some(upper) = *max_len
                    && chars > upper
                {
                    return Err(ValidationError::new(format!(
                        "string length {chars} > max_len {upper}"
                    )));
                }
                if let Some(regex_text) = pattern {
                    let regex = regex::Regex::new(regex_text).map_err(|err| {
                        ValidationError::new(format!("schema pattern is invalid: {err}"))
                    })?;
                    if !regex.is_match(text) {
                        return Err(ValidationError::new(format!(
                            "string did not match pattern /{regex_text}/"
                        )));
                    }
                }
                if let Some(declared_format) = format {
                    validate_string_format(*declared_format, text)?;
                }
                Ok(())
            }
            Self::Bytes { max_len } => {
                let text = value
                    .as_str()
                    .ok_or_else(|| ValidationError::new("expected base64 string for bytes"))?;
                if let Some(upper) = *max_len
                    && text.len() > upper
                {
                    return Err(ValidationError::new(format!(
                        "bytes length {len} > max_len {upper}",
                        len = text.len()
                    )));
                }
                Ok(())
            }
            Self::Seq {
                items,
                min_items,
                max_items,
            } => {
                let array = value
                    .as_array()
                    .ok_or_else(|| ValidationError::new("expected array"))?;
                if let Some(lower) = *min_items
                    && array.len() < lower
                {
                    return Err(ValidationError::new(format!(
                        "array length {len} < min_items {lower}",
                        len = array.len()
                    )));
                }
                if let Some(upper) = *max_items
                    && array.len() > upper
                {
                    return Err(ValidationError::new(format!(
                        "array length {len} > max_items {upper}",
                        len = array.len()
                    )));
                }
                for (index, element) in array.iter().enumerate() {
                    items
                        .validate_inner(element, resolver, visiting)
                        .map_err(|err| err.at_index(index))?;
                }
                Ok(())
            }
            Self::Map { keys: _, values } => {
                let object = value
                    .as_object()
                    .ok_or_else(|| ValidationError::new("expected object"))?;
                // Map.keys is always String in JSON; we honour the values
                // schema for each entry. Map.keys is preserved for emitters.
                for (name, child) in object {
                    values
                        .validate_inner(child, resolver, visiting)
                        .map_err(|err| err.at_field(name))?;
                }
                Ok(())
            }
            Self::Tuple(parts) => {
                let array = value
                    .as_array()
                    .ok_or_else(|| ValidationError::new("expected array for tuple"))?;
                if array.len() != parts.len() {
                    return Err(ValidationError::new(format!(
                        "tuple length {len} != {expected}",
                        len = array.len(),
                        expected = parts.len()
                    )));
                }
                for (index, (element, part_schema)) in array.iter().zip(parts.iter()).enumerate() {
                    part_schema
                        .validate_inner(element, resolver, visiting)
                        .map_err(|err| err.at_index(index))?;
                }
                Ok(())
            }
            Self::Struct { name: _, fields } => {
                let object = value
                    .as_object()
                    .ok_or_else(|| ValidationError::new("expected object"))?;
                for field in fields {
                    match object.get(&field.name) {
                        Some(child) => {
                            field
                                .schema
                                .validate_inner(child, resolver, visiting)
                                .map_err(|err| err.at_field(&field.name))?;
                        }
                        None => {
                            if !field.flags.optional {
                                return Err(ValidationError::new(format!(
                                    "missing required field `{name}`",
                                    name = field.name
                                )));
                            }
                        }
                    }
                }
                Ok(())
            }
            Self::Enum { name: _, variants } => {
                // unit variant: top-level value is the variant name string.
                if let Some(text) = value.as_str()
                    && variants
                        .iter()
                        .any(|variant| variant.payload.is_none() && variant.name == text)
                {
                    return Ok(());
                }
                // payload variant: { "<variant>": <payload> }
                if let Some(object) = value.as_object() {
                    for (key, payload) in object {
                        if let Some(variant) = variants.iter().find(|variant| variant.name == *key)
                        {
                            return match &variant.payload {
                                Some(inner) => inner
                                    .validate_inner(payload, resolver, visiting)
                                    .map_err(|err| err.at_field(key)),
                                None => Err(ValidationError::new(format!(
                                    "variant `{key}` is a unit, not a payload variant"
                                ))),
                            };
                        }
                    }
                }
                Err(ValidationError::new("no enum variant matched"))
            }
            Self::Optional(inner) => {
                if value.is_null() {
                    return Ok(());
                }
                inner.validate_inner(value, resolver, visiting)
            }
            Self::Ref(name) => {
                if visiting.iter().any(|seen| seen == name) {
                    return Err(ValidationError::new(format!(
                        "cycle detected through schema ref `{name}`"
                    )));
                }
                let resolved = resolver
                    .resolve(name)
                    .ok_or_else(|| ValidationError::new(format!("unknown schema ref `{name}`")))?;
                visiting.push(name.clone());
                let outcome = resolved.validate_inner(value, resolver, visiting);
                visiting.pop();
                outcome
            }
            Self::Any => Ok(()),
        }
    }
}

#[cfg(feature = "schema")]
fn validate_string_format(format: StringFormat, text: &str) -> Result<(), ValidationError> {
    match format {
        StringFormat::Email => {
            // bare structural check: one `@` with non-empty sides
            let (local, domain) = text
                .split_once('@')
                .ok_or_else(|| ValidationError::new("expected email — missing `@`"))?;
            if local.is_empty() || domain.is_empty() || !domain.contains('.') {
                return Err(ValidationError::new("expected email — malformed"));
            }
        }
        StringFormat::Url => {
            if url::Url::parse(text).is_err() {
                return Err(ValidationError::new("expected url — failed to parse"));
            }
        }
        StringFormat::Uuid => {
            // 8-4-4-4-12 hex with hyphens
            if text.len() != 36 {
                return Err(ValidationError::new("expected uuid — wrong length"));
            }
            let segments: Vec<&str> = text.split('-').collect();
            if segments.len() != 5
                || segments[0].len() != 8
                || segments[1].len() != 4
                || segments[2].len() != 4
                || segments[3].len() != 4
                || segments[4].len() != 12
                || !text
                    .chars()
                    .all(|character| character == '-' || character.is_ascii_hexdigit())
            {
                return Err(ValidationError::new("expected uuid — malformed"));
            }
        }
        StringFormat::Ipv4 => {
            if text.parse::<core::net::Ipv4Addr>().is_err() {
                return Err(ValidationError::new("expected ipv4 — failed to parse"));
            }
        }
        StringFormat::Ipv6 => {
            if text.parse::<core::net::Ipv6Addr>().is_err() {
                return Err(ValidationError::new("expected ipv6 — failed to parse"));
            }
        }
        StringFormat::Date => {
            if !text
                .chars()
                .enumerate()
                .all(|(index, character)| match index {
                    4 | 7 => character == '-',
                    _ => character.is_ascii_digit(),
                })
                || text.len() != 10
            {
                return Err(ValidationError::new("expected date YYYY-MM-DD"));
            }
        }
        StringFormat::DateTime => {
            if time::OffsetDateTime::parse(text, &time::format_description::well_known::Rfc3339)
                .is_err()
            {
                return Err(ValidationError::new("expected RFC3339 datetime"));
            }
        }
        StringFormat::Hostname => {
            if text.is_empty()
                || text.len() > 253
                || text.starts_with('-')
                || text.ends_with('-')
                || text.split('.').any(|label| {
                    label.is_empty()
                        || label.len() > 63
                        || !label
                            .chars()
                            .all(|character| character.is_ascii_alphanumeric() || character == '-')
                })
            {
                return Err(ValidationError::new("expected hostname"));
            }
        }
    }
    Ok(())
}

/// Named-schema registry. Mirrors the other proxima registries.
/// Lock-free via `ArcSwap<BTreeMap<...>>` — register is CAS-loop CoW,
/// `get` / `snapshot` are atomic loads.
#[cfg(feature = "schema-std")]
pub struct SchemaRegistry {
    schemas: ArcSwap<BTreeMap<String, Schema>>,
}

#[cfg(feature = "schema-std")]
impl Default for SchemaRegistry {
    fn default() -> Self {
        Self {
            schemas: ArcSwap::from_pointee(BTreeMap::new()),
        }
    }
}

#[cfg(feature = "schema-std")]
impl SchemaRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, name: impl Into<String>, schema: Schema) -> Result<(), ProximaError> {
        let name = name.into();
        loop {
            let current = self.schemas.load_full();
            if current.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "schema `{name}` already registered"
                )));
            }
            let mut next: BTreeMap<String, Schema> = (*current).clone();
            next.insert(name.clone(), schema.clone());
            let prev = self.schemas.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<Schema> {
        self.schemas.load_full().get(name).cloned()
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.schemas.load_full().keys().cloned().collect()
    }

    #[must_use]
    pub fn snapshot(&self) -> BTreeMap<String, Schema> {
        (*self.schemas.load_full()).clone()
    }
}

#[cfg(feature = "schema-std")]
impl SchemaResolver for SchemaRegistry {
    fn resolve(&self, name: &str) -> Option<Schema> {
        self.get(name)
    }
}

#[cfg(feature = "schema-std")]
impl SchemaResolver for Arc<SchemaRegistry> {
    fn resolve(&self, name: &str) -> Option<Schema> {
        SchemaRegistry::get(self, name)
    }
}

#[cfg(all(test, feature = "schema"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use serde_json::json;

    fn user_struct() -> Schema {
        Schema::Struct {
            name: "User".into(),
            fields: vec![
                StructField {
                    name: "name".into(),
                    schema: Schema::String {
                        pattern: None,
                        format: None,
                        min_len: Some(1),
                        max_len: Some(64),
                    },
                    flags: FieldFlags::default(),
                },
                StructField {
                    name: "age".into(),
                    schema: Schema::UInt {
                        min: Some(0),
                        max: Some(150),
                    },
                    flags: FieldFlags {
                        optional: true,
                        ..FieldFlags::default()
                    },
                },
            ],
        }
    }

    #[test]
    fn struct_accepts_valid_instance() {
        let schema = user_struct();
        let value = json!({"name": "brian", "age": 35});
        schema.validate(&value, &EmptyResolver).expect("valid");
    }

    #[test]
    fn struct_accepts_missing_optional_field() {
        let schema = user_struct();
        let value = json!({"name": "brian"});
        schema.validate(&value, &EmptyResolver).expect("valid");
    }

    #[test]
    fn struct_rejects_missing_required_field() {
        let schema = user_struct();
        let value = json!({"age": 30});
        let err = schema
            .validate(&value, &EmptyResolver)
            .expect_err("invalid");
        assert!(err.message.contains("missing required field `name`"));
    }

    #[test]
    fn string_rejects_too_short() {
        let schema = Schema::String {
            pattern: None,
            format: None,
            min_len: Some(3),
            max_len: None,
        };
        let err = schema
            .validate(&json!("ab"), &EmptyResolver)
            .expect_err("short");
        assert!(err.message.contains("min_len"));
    }

    #[test]
    fn string_pattern_enforced() {
        let schema = Schema::String {
            pattern: Some("^[a-z]+$".into()),
            format: None,
            min_len: None,
            max_len: None,
        };
        schema
            .validate(&json!("abc"), &EmptyResolver)
            .expect("matches");
        schema
            .validate(&json!("abc1"), &EmptyResolver)
            .expect_err("no digit allowed");
    }

    #[test]
    fn int_bounds_enforced() {
        let schema = Schema::Int {
            min: Some(0),
            max: Some(10),
        };
        schema.validate(&json!(5), &EmptyResolver).expect("inside");
        schema
            .validate(&json!(-1), &EmptyResolver)
            .expect_err("below");
        schema
            .validate(&json!(11), &EmptyResolver)
            .expect_err("above");
    }

    #[test]
    fn seq_bounds_and_item_schema() {
        let schema = Schema::Seq {
            items: Box::new(Schema::UInt {
                min: None,
                max: None,
            }),
            min_items: Some(1),
            max_items: Some(3),
        };
        schema
            .validate(&json!([1, 2, 3]), &EmptyResolver)
            .expect("inside");
        schema
            .validate(&json!([]), &EmptyResolver)
            .expect_err("empty");
        schema
            .validate(&json!([1, 2, 3, 4]), &EmptyResolver)
            .expect_err("too many");
        let err = schema
            .validate(&json!([1, "not-a-number"]), &EmptyResolver)
            .expect_err("bad item");
        assert_eq!(err.path_string(), "$[1]");
    }

    #[test]
    fn enum_unit_and_payload_variants() {
        let schema = Schema::Enum {
            name: "Status".into(),
            variants: vec![
                EnumVariant {
                    name: "ok".into(),
                    payload: None,
                },
                EnumVariant {
                    name: "failed".into(),
                    payload: Some(Box::new(Schema::String {
                        pattern: None,
                        format: None,
                        min_len: None,
                        max_len: None,
                    })),
                },
            ],
        };
        schema.validate(&json!("ok"), &EmptyResolver).expect("unit");
        schema
            .validate(&json!({"failed": "disk full"}), &EmptyResolver)
            .expect("payload");
        schema
            .validate(&json!("unknown"), &EmptyResolver)
            .expect_err("no match");
    }

    #[test]
    fn optional_accepts_null() {
        let schema = Schema::Optional(Box::new(Schema::Bool));
        schema.validate(&Value::Null, &EmptyResolver).expect("null");
        schema
            .validate(&json!(true), &EmptyResolver)
            .expect("inner");
    }

    #[cfg(feature = "schema-std")]
    #[test]
    fn ref_resolves_through_registry() {
        let registry = SchemaRegistry::new();
        registry.register("User", user_struct()).expect("register");
        let outer = Schema::Struct {
            name: "Request".into(),
            fields: vec![StructField {
                name: "user".into(),
                schema: Schema::Ref("User".into()),
                flags: FieldFlags::default(),
            }],
        };
        outer
            .validate(&json!({"user": {"name": "brian"}}), &registry)
            .expect("valid through ref");
    }

    #[cfg(feature = "schema-std")]
    #[test]
    fn ref_cycle_detected() {
        let registry = SchemaRegistry::new();
        registry
            .register(
                "Cycle",
                Schema::Struct {
                    name: "Cycle".into(),
                    fields: vec![StructField {
                        name: "next".into(),
                        schema: Schema::Ref("Cycle".into()),
                        flags: FieldFlags::default(),
                    }],
                },
            )
            .expect("register");
        let err = Schema::Ref("Cycle".into())
            .validate(&json!({"next": {"next": {}}}), &registry)
            .expect_err("expected cycle error");
        assert!(err.message.contains("cycle detected"));
    }

    #[test]
    fn string_format_email_accepts_and_rejects() {
        let schema = Schema::String {
            pattern: None,
            format: Some(StringFormat::Email),
            min_len: None,
            max_len: None,
        };
        schema
            .validate(&json!("brian@example.com"), &EmptyResolver)
            .expect("valid email");
        schema
            .validate(&json!("not-an-email"), &EmptyResolver)
            .expect_err("missing @");
    }

    #[test]
    fn string_format_uuid_round_trip() {
        let schema = Schema::String {
            pattern: None,
            format: Some(StringFormat::Uuid),
            min_len: None,
            max_len: None,
        };
        schema
            .validate(
                &json!("550e8400-e29b-41d4-a716-446655440000"),
                &EmptyResolver,
            )
            .expect("valid uuid");
        schema
            .validate(&json!("not-a-uuid"), &EmptyResolver)
            .expect_err("malformed uuid");
    }

    #[test]
    fn validation_error_path_string() {
        let err = ValidationError::new("bad")
            .at_field("name")
            .at_field("user")
            .at_index(3);
        assert_eq!(err.path_string(), "$[3].user.name");
    }

    #[cfg(feature = "schema-std")]
    #[test]
    fn registry_rejects_duplicate_registration() {
        let registry = SchemaRegistry::new();
        registry.register("X", Schema::Bool).expect("first");
        assert!(registry.register("X", Schema::Bool).is_err());
    }

    #[test]
    fn schema_round_trips_through_serde_json() {
        let schema = user_struct();
        let encoded = serde_json::to_value(&schema).expect("encode");
        let decoded: Schema = serde_json::from_value(encoded).expect("decode");
        decoded
            .validate(&json!({"name": "brian"}), &EmptyResolver)
            .expect("decoded schema still validates");
    }
}

#[cfg(all(test, feature = "schema"))]
mod round_trip_tests {
    use super::*;
    use alloc::vec;

    fn open_string() -> Schema {
        Schema::String {
            pattern: None,
            format: None,
            min_len: None,
            max_len: None,
        }
    }

    fn sample() -> Schema {
        Schema::Struct {
            name: "Note".to_string(),
            fields: vec![
                StructField {
                    name: "id".to_string(),
                    schema: open_string(),
                    flags: FieldFlags::default(),
                },
                StructField {
                    name: "tags".to_string(),
                    schema: Schema::Seq {
                        items: Box::new(open_string()),
                        min_items: None,
                        max_items: None,
                    },
                    flags: FieldFlags::default(),
                },
                StructField {
                    name: "score".to_string(),
                    schema: Schema::Optional(Box::new(Schema::Float { finite: false })),
                    flags: FieldFlags {
                        optional: true,
                        ..FieldFlags::default()
                    },
                },
            ],
        }
    }

    fn to_json(schema: &Schema) -> String {
        match serde_json::to_string(schema) {
            Ok(text) => text,
            Err(err) => panic!("json serialize failed: {err}"),
        }
    }

    #[test]
    fn schema_round_trips_through_json() {
        let original = sample();
        let json = to_json(&original);
        let back: Schema = match serde_json::from_str(&json) {
            Ok(value) => value,
            Err(err) => panic!("json deserialize failed: {err}"),
        };
        assert_eq!(json, to_json(&back), "json round-trip is stable");
    }

    #[cfg(feature = "schema-std")]
    #[test]
    fn schema_round_trips_through_toml() {
        let original = sample();
        let toml_text = match toml::to_string(&original) {
            Ok(text) => text,
            Err(err) => panic!("toml serialize failed: {err}"),
        };
        let back: Schema = match toml::from_str(&toml_text) {
            Ok(value) => value,
            Err(err) => panic!("toml deserialize failed: {err}\n--- toml was ---\n{toml_text}"),
        };
        assert_eq!(
            to_json(&original),
            to_json(&back),
            "toml round-trip preserves the schema"
        );
    }

    // use serde-PRODUCED json (well-formed, includes the `value` content) and
    // only vary the tag, so the test measures the alias and nothing else.
    #[test]
    fn canonical_seq_loads_and_array_alias() {
        let seq = Schema::Seq {
            items: Box::new(open_string()),
            min_items: None,
            max_items: None,
        };
        let canonical = to_json(&seq);
        assert!(
            matches!(
                serde_json::from_str::<Schema>(&canonical),
                Ok(Schema::Seq { .. })
            ),
            "canonical seq loads: {canonical}"
        );

        // swap only the tag: does the json-schema `array` spelling alias to Seq?
        let array_form = canonical.replacen("\"seq\"", "\"array\"", 1);
        let parsed = serde_json::from_str::<Schema>(&array_form);
        assert!(
            matches!(parsed, Ok(Schema::Seq { .. })),
            "the `array` alias loads as Seq, got {parsed:?} from {array_form}"
        );
    }
}
