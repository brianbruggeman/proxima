//! Emit JSON Schema 2020-12 from a `Schema`. Refs become `$ref`s into
//! the shared `$defs` section; named structs/enums also surface there.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use serde_json::{Map, Value, json};

use crate::schema::{EnumVariant, Schema, StringFormat, StructField};

/// Emit a single schema, with optional named referenced schemas placed in
/// `$defs`. Returns a JSON Schema 2020-12 document.
#[must_use]
pub fn emit(root: &Schema, refs: &BTreeMap<String, Schema>) -> Value {
    let mut definitions = Map::new();
    for (name, schema) in refs {
        definitions.insert(name.clone(), schema_to_value(schema));
    }
    let mut value = schema_to_value(root);
    if !definitions.is_empty()
        && let Value::Object(object) = &mut value
    {
        object.insert("$defs".into(), Value::Object(definitions));
    }
    if let Value::Object(object) = &mut value {
        object.insert(
            "$schema".into(),
            Value::String("https://json-schema.org/draft/2020-12/schema".into()),
        );
    }
    value
}

fn schema_to_value(schema: &Schema) -> Value {
    match schema {
        Schema::Bool => json!({ "type": "boolean" }),
        Schema::Int { min, max } => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("integer".into()));
            if let Some(lower) = min {
                object.insert("minimum".into(), Value::Number((*lower).into()));
            }
            if let Some(upper) = max {
                object.insert("maximum".into(), Value::Number((*upper).into()));
            }
            Value::Object(object)
        }
        Schema::UInt { min, max } => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("integer".into()));
            object.insert("minimum".into(), Value::Number(min.unwrap_or(0).into()));
            if let Some(upper) = max {
                object.insert("maximum".into(), Value::Number((*upper).into()));
            }
            Value::Object(object)
        }
        Schema::Float { finite: _ } => json!({ "type": "number" }),
        Schema::String {
            pattern,
            format,
            min_len,
            max_len,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("string".into()));
            if let Some(text) = pattern {
                object.insert("pattern".into(), Value::String(text.clone()));
            }
            if let Some(declared) = format {
                object.insert(
                    "format".into(),
                    Value::String(string_format_name(*declared).into()),
                );
            }
            if let Some(lower) = min_len {
                object.insert("minLength".into(), Value::Number((*lower).into()));
            }
            if let Some(upper) = max_len {
                object.insert("maxLength".into(), Value::Number((*upper).into()));
            }
            Value::Object(object)
        }
        Schema::Bytes { max_len } => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("string".into()));
            object.insert("contentEncoding".into(), Value::String("base64".into()));
            if let Some(upper) = max_len {
                object.insert("maxLength".into(), Value::Number((*upper).into()));
            }
            Value::Object(object)
        }
        Schema::Seq {
            items,
            min_items,
            max_items,
        } => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("array".into()));
            object.insert("items".into(), schema_to_value(items));
            if let Some(lower) = min_items {
                object.insert("minItems".into(), Value::Number((*lower).into()));
            }
            if let Some(upper) = max_items {
                object.insert("maxItems".into(), Value::Number((*upper).into()));
            }
            Value::Object(object)
        }
        Schema::Map { keys: _, values } => json!({
            "type": "object",
            "additionalProperties": schema_to_value(values),
        }),
        Schema::Tuple(parts) => {
            let items: Vec<Value> = parts.iter().map(schema_to_value).collect();
            let count = items.len();
            let value_count = Value::Number(count.into());
            json!({
                "type": "array",
                "prefixItems": items,
                "minItems": value_count.clone(),
                "maxItems": value_count,
            })
        }
        Schema::Struct { name: _, fields } => emit_struct(fields),
        Schema::Enum { name: _, variants } => emit_enum(variants),
        Schema::Optional(inner) => {
            let mut object = Map::new();
            object.insert(
                "oneOf".into(),
                Value::Array(vec![schema_to_value(inner), json!({ "type": "null" })]),
            );
            Value::Object(object)
        }
        Schema::Ref(name) => json!({ "$ref": format!("#/$defs/{name}") }),
        Schema::Any => Value::Object(Map::new()),
    }
}

fn emit_struct(fields: &[StructField]) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for field in fields {
        properties.insert(field.name.clone(), schema_to_value(&field.schema));
        if !field.flags.optional {
            required.push(Value::String(field.name.clone()));
        }
    }
    let mut object = Map::new();
    object.insert("type".into(), Value::String("object".into()));
    object.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        object.insert("required".into(), Value::Array(required));
    }
    object.insert("additionalProperties".into(), Value::Bool(false));
    Value::Object(object)
}

fn emit_enum(variants: &[EnumVariant]) -> Value {
    let mut one_of = Vec::with_capacity(variants.len());
    for variant in variants {
        match &variant.payload {
            None => one_of.push(json!({
                "const": variant.name,
            })),
            Some(payload) => {
                let mut object = Map::new();
                object.insert("type".into(), Value::String("object".into()));
                let mut properties = Map::new();
                properties.insert(variant.name.clone(), schema_to_value(payload));
                object.insert("properties".into(), Value::Object(properties));
                object.insert(
                    "required".into(),
                    Value::Array(vec![Value::String(variant.name.clone())]),
                );
                object.insert("additionalProperties".into(), Value::Bool(false));
                one_of.push(Value::Object(object));
            }
        }
    }
    json!({ "oneOf": one_of })
}

fn string_format_name(format: StringFormat) -> &'static str {
    match format {
        StringFormat::Email => "email",
        StringFormat::Url => "uri",
        StringFormat::Uuid => "uuid",
        StringFormat::Ipv4 => "ipv4",
        StringFormat::Ipv6 => "ipv6",
        StringFormat::Date => "date",
        StringFormat::DateTime => "date-time",
        StringFormat::Hostname => "hostname",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::FieldFlags;
    use alloc::boxed::Box;

    #[test]
    fn emits_struct_with_required_and_optional() {
        let schema = Schema::Struct {
            name: "User".into(),
            fields: vec![
                StructField {
                    name: "name".into(),
                    schema: Schema::String {
                        pattern: None,
                        format: None,
                        min_len: Some(1),
                        max_len: None,
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
        };
        let emitted = emit(&schema, &BTreeMap::new());
        assert_eq!(emitted["type"], "object");
        assert_eq!(emitted["properties"]["name"]["type"], "string");
        assert_eq!(emitted["properties"]["name"]["minLength"], 1);
        assert_eq!(emitted["properties"]["age"]["type"], "integer");
        assert_eq!(emitted["properties"]["age"]["minimum"], 0);
        assert_eq!(emitted["properties"]["age"]["maximum"], 150);
        assert_eq!(emitted["required"], serde_json::json!(["name"]));
        assert_eq!(emitted["additionalProperties"], false);
        assert_eq!(
            emitted["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }

    #[test]
    fn emits_ref_into_defs() {
        let mut refs = BTreeMap::new();
        refs.insert("User".into(), Schema::Bool);
        let root = Schema::Ref("User".into());
        let emitted = emit(&root, &refs);
        assert_eq!(emitted["$ref"], "#/$defs/User");
        assert_eq!(emitted["$defs"]["User"]["type"], "boolean");
    }

    #[test]
    fn emits_seq_with_bounds() {
        let schema = Schema::Seq {
            items: Box::new(Schema::String {
                pattern: None,
                format: None,
                min_len: None,
                max_len: None,
            }),
            min_items: Some(1),
            max_items: Some(10),
        };
        let emitted = emit(&schema, &BTreeMap::new());
        assert_eq!(emitted["type"], "array");
        assert_eq!(emitted["minItems"], 1);
        assert_eq!(emitted["maxItems"], 10);
        assert_eq!(emitted["items"]["type"], "string");
    }

    #[test]
    fn emits_optional_as_one_of() {
        let schema = Schema::Optional(Box::new(Schema::Bool));
        let emitted = emit(&schema, &BTreeMap::new());
        let one_of = emitted["oneOf"].as_array().expect("oneOf array");
        assert_eq!(one_of.len(), 2);
    }

    #[test]
    fn emits_enum_unit_and_payload_variants() {
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
        let emitted = emit(&schema, &BTreeMap::new());
        let one_of = emitted["oneOf"].as_array().expect("oneOf array");
        assert_eq!(one_of.len(), 2);
        assert_eq!(one_of[0]["const"], "ok");
        assert_eq!(one_of[1]["properties"]["failed"]["type"], "string");
    }

    #[test]
    fn emits_string_format_names() {
        let schema = Schema::String {
            pattern: None,
            format: Some(StringFormat::Email),
            min_len: None,
            max_len: None,
        };
        let emitted = emit(&schema, &BTreeMap::new());
        assert_eq!(emitted["format"], "email");
    }
}
