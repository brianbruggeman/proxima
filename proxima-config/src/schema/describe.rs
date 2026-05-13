//! `Describe`: types that generate their own [`Schema`] IR from the type itself,
//! so the contract cannot drift from the Rust shape. derive it with
//! `#[derive(Schema)]` (the `schema-derive` feature) the way serde derives are
//! used; the std impls below bottom out the recursion.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::schema::FieldFlags;
use crate::schema::Schema;
use crate::schema::StructField;

/// a type whose value-shape is known at compile time. `schema()` returns the IR
/// the `validate` middleware and the `describe` emitters consume.
pub trait Describe {
    fn schema() -> Schema;
}

macro_rules! describe_signed {
    ($($ty:ty),*) => { $(impl Describe for $ty { fn schema() -> Schema { Schema::Int { min: None, max: None } } })* };
}
describe_signed!(i8, i16, i32, i64, isize);

macro_rules! describe_unsigned {
    ($($ty:ty),*) => { $(impl Describe for $ty { fn schema() -> Schema { Schema::UInt { min: None, max: None } } })* };
}
describe_unsigned!(u8, u16, u32, u64, usize);

impl Describe for f32 {
    fn schema() -> Schema {
        Schema::Float { finite: false }
    }
}

impl Describe for f64 {
    fn schema() -> Schema {
        Schema::Float { finite: false }
    }
}

impl Describe for bool {
    fn schema() -> Schema {
        Schema::Bool
    }
}

fn open_string() -> Schema {
    Schema::String {
        pattern: None,
        format: None,
        min_len: None,
        max_len: None,
    }
}

impl Describe for String {
    fn schema() -> Schema {
        open_string()
    }
}

impl<T: Describe> Describe for Vec<T> {
    fn schema() -> Schema {
        Schema::Seq {
            items: Box::new(T::schema()),
            min_items: None,
            max_items: None,
        }
    }
}

impl<T: Describe, const N: usize> Describe for [T; N] {
    fn schema() -> Schema {
        Schema::Seq {
            items: Box::new(T::schema()),
            min_items: Some(N),
            max_items: Some(N),
        }
    }
}

impl<T: Describe> Describe for Option<T> {
    fn schema() -> Schema {
        Schema::Optional(Box::new(T::schema()))
    }
}

impl<T: Describe> Describe for Box<T> {
    fn schema() -> Schema {
        T::schema()
    }
}

impl<K: Describe, V: Describe> Describe for BTreeMap<K, V> {
    fn schema() -> Schema {
        Schema::Map {
            keys: Box::new(K::schema()),
            values: Box::new(V::schema()),
        }
    }
}

/// arbitrary serde value — the format-neutral escape hatch for genuinely
/// free-form payloads (use a typed struct whenever the shape is known).
/// `serde-value` has no no_std support, so this escape hatch is std-only.
#[cfg(feature = "schema-std")]
impl Describe for serde_value::Value {
    fn schema() -> Schema {
        Schema::Any
    }
}

/// build a `Struct` field; the derive emits one call per field so the field list
/// stays a single mechanical mapping from the type.
#[must_use]
pub fn field(name: &str, schema: Schema, optional: bool) -> StructField {
    StructField {
        name: name.to_string(),
        schema,
        flags: FieldFlags {
            optional,
            ..FieldFlags::default()
        },
    }
}

#[cfg(all(test, feature = "schema-derive"))]
mod tests {
    use alloc::collections::BTreeMap;

    use crate::schema::{Describe, EmptyResolver, Schema};

    // the type IS the contract: derive the schema, then validate a typed value
    // of that same type against it — format-neutral, no json type in sight.
    #[derive(serde::Serialize, Schema)]
    struct Wire {
        id: String,
        count: u64,
        tags: Vec<String>,
        note: Option<String>,
    }

    #[test]
    fn typed_value_validates_against_its_derived_schema() {
        let value = Wire {
            id: "x".to_string(),
            count: 3,
            tags: vec!["a".to_string()],
            note: None,
        };
        let schema = Wire::schema();
        assert!(
            schema.validate(&value, &EmptyResolver).is_ok(),
            "a typed value validates against its own derived schema"
        );

        // the same generic validate accepts a free-form value, and rejects a
        // wrong-typed one — proving it is value-type-agnostic, not json-bound.
        let wrong = serde_json::json!({ "id": 5, "count": 3, "tags": [], "note": null });
        assert!(
            schema.validate(&wrong, &EmptyResolver).is_err(),
            "a wrong-typed value is rejected"
        );
    }

    #[derive(Schema)]
    #[allow(dead_code)]
    struct Note {
        id: String,
        count: u64,
        ratio: f64,
        tags: Vec<String>,
        score: Option<f64>,
        #[schema(rename = "type")]
        kind: String,
        #[schema(skip)]
        internal: bool,
    }

    #[test]
    fn schema_is_generated_from_the_struct() {
        let Schema::Struct { name, fields } = Note::schema() else {
            panic!("expected a struct schema");
        };
        assert_eq!(name, "Note");
        assert_eq!(fields.len(), 6, "the skipped field is omitted");
        assert_eq!(fields[0].name, "id");
        assert!(matches!(fields[0].schema, Schema::String { .. }));
        assert!(matches!(fields[1].schema, Schema::UInt { .. }));
        assert!(matches!(fields[2].schema, Schema::Float { .. }));
        assert!(matches!(fields[3].schema, Schema::Seq { .. }));
        assert_eq!(fields[4].name, "score");
        assert!(
            fields[4].flags.optional,
            "Option fields are marked optional"
        );
        assert!(matches!(fields[4].schema, Schema::Optional(_)));
        assert_eq!(fields[5].name, "type", "schema rename tracks the wire name");
    }

    #[test]
    fn generated_schema_validates_real_values() {
        let schema = Note::schema();
        let good = serde_json::json!({ "id": "x", "count": 3, "ratio": 0.5, "tags": ["a"], "score": null, "type": "concept" });
        assert!(
            schema.validate(&good, &EmptyResolver).is_ok(),
            "a conforming value validates"
        );

        let wrong_type =
            serde_json::json!({ "id": 5, "count": 3, "ratio": 0.5, "tags": [], "type": "concept" });
        assert!(
            schema.validate(&wrong_type, &EmptyResolver).is_err(),
            "a wrong-typed field is rejected"
        );

        let missing_required =
            serde_json::json!({ "id": "x", "ratio": 0.5, "tags": [], "type": "concept" });
        assert!(
            schema.validate(&missing_required, &EmptyResolver).is_err(),
            "a missing required field is rejected"
        );
    }

    // a fixed-size array is a bounded sequence — exactly N items, no fewer, no
    // more. real DTO shape: latency percentiles `[p50, p95, p99]` as `[f64; 3]`.
    #[test]
    fn fixed_array_is_a_bounded_seq() {
        let Schema::Seq {
            items,
            min_items,
            max_items,
        } = <[f64; 3]>::schema()
        else {
            panic!("expected a seq schema for a fixed array");
        };
        assert!(matches!(*items, Schema::Float { .. }));
        assert_eq!(
            (min_items, max_items),
            (Some(3), Some(3)),
            "exactly N items"
        );
        let schema = <[f64; 3]>::schema();
        assert!(
            schema
                .validate(&serde_json::json!([100.0, 200.0, 300.0]), &EmptyResolver)
                .is_ok()
        );
        assert!(
            schema
                .validate(&serde_json::json!([100.0, 200.0]), &EmptyResolver)
                .is_err(),
            "too few items is rejected"
        );
    }

    // a `#[serde(default)]` field is absent-tolerant on the wire, so the schema
    // must mark it optional — otherwise validation would reject a legitimate
    // response that serde would happily default. real DTO shape: the daemon omits
    // empty fields and the client fills them via serde default.
    #[derive(serde::Deserialize, Schema)]
    #[allow(dead_code)]
    struct WireDefaults {
        id: String,
        #[serde(default)]
        content: String,
        #[serde(default = "one")]
        count: u64,
        maybe: Option<String>,
    }

    fn one() -> u64 {
        1
    }

    #[test]
    fn serde_default_fields_are_optional() {
        let Schema::Struct { fields, .. } = WireDefaults::schema() else {
            panic!("expected a struct schema");
        };
        assert!(
            !fields[0].flags.optional,
            "a plain required field stays required"
        );
        assert!(
            fields[1].flags.optional,
            "#[serde(default)] is absent-tolerant"
        );
        assert!(
            fields[2].flags.optional,
            "#[serde(default = \"...\")] is absent-tolerant"
        );
        assert!(fields[3].flags.optional, "Option is absent-tolerant");

        let schema = WireDefaults::schema();
        let omitting_defaults = serde_json::json!({ "id": "x" });
        assert!(
            schema.validate(&omitting_defaults, &EmptyResolver).is_ok(),
            "a response omitting default fields validates"
        );
        let omitting_required = serde_json::json!({ "content": "hi" });
        assert!(
            schema.validate(&omitting_required, &EmptyResolver).is_err(),
            "omitting the required id is rejected"
        );
    }

    // the schema field name must be the WIRE name, so `#[serde(rename = "...")]`
    // is tracked (else validation rejects a real response keyed by the wire name).
    // real DTO shape: a short wire key (`wTok`) behind a descriptive Rust field.
    #[derive(serde::Deserialize, Schema)]
    #[allow(dead_code)]
    struct Renamed {
        #[serde(rename = "wTok")]
        wasted_tokens: u64,
        #[serde(rename = "type")]
        #[schema(rename = "kind")]
        category: String,
    }

    #[test]
    fn serde_rename_sets_the_wire_name() {
        let Schema::Struct { fields, .. } = Renamed::schema() else {
            panic!("expected a struct schema");
        };
        assert_eq!(
            fields[0].name, "wTok",
            "serde rename is the schema field name"
        );
        assert_eq!(
            fields[1].name, "kind",
            "schema rename overrides serde rename"
        );
    }

    // the keystone: a derived type dumps openapi with zero hand-authoring —
    // `#[derive(Schema)]` -> Schema -> emit_openapi -> contract document.
    #[test]
    fn derived_schema_dumps_openapi() {
        let mut schemas = BTreeMap::new();
        schemas.insert("Note".to_string(), Note::schema());
        let doc = crate::schema::emit::emit_openapi("example-service", "1.0", &schemas, serde_json::json!({}));

        assert_eq!(doc["openapi"], "3.1.0");
        let note = &doc["components"]["schemas"]["Note"];
        assert_eq!(
            note["type"], "object",
            "the struct emits an object schema, got {doc}"
        );
        assert_eq!(note["properties"]["id"]["type"], "string");
        assert_eq!(note["properties"]["count"]["type"], "integer");
        assert_eq!(note["properties"]["tags"]["type"], "array");
        let Some(required) = note["required"].as_array() else {
            panic!("openapi component has a required array, got {doc}");
        };
        assert!(required.iter().any(|name| name == "id"), "id is required");
        assert!(
            !required.iter().any(|name| name == "score"),
            "the Option field is not required"
        );
    }
}
