//! Emit OpenAPI 3.1 from a registry of named schemas. OpenAPI 3.1
//! aligns with JSON Schema 2020-12 so component schemas reuse the
//! json_schema emitter; this layer wraps it in the OpenAPI envelope.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;

use serde_json::{Map, Value, json};

use crate::schema::Schema;
use crate::schema::emit::json_schema;

/// Build a minimal OpenAPI 3.1 document with every registered schema
/// surfaced in `components.schemas`. `info.title` and `info.version`
/// come from the caller; `paths` may be empty when called purely for
/// schema export.
#[must_use]
pub fn emit(title: &str, version: &str, schemas: &BTreeMap<String, Schema>, paths: Value) -> Value {
    let mut components_schemas = Map::new();
    for (name, schema) in schemas {
        // each schema is emitted in isolation; cross-refs are encoded
        // as $ref pointers and openapi resolves them within components.
        let body = json_schema::emit(schema, &BTreeMap::new());
        components_schemas.insert(name.clone(), rewrite_defs_refs_to_components(body));
    }
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": title,
            "version": version,
        },
        "paths": paths,
        "components": {
            "schemas": Value::Object(components_schemas),
        },
    })
}

/// Walk a `$ref` graph rewriting `#/$defs/X` → `#/components/schemas/X`.
/// The json_schema emitter places refs under `$defs`; OpenAPI uses
/// `components/schemas`.
fn rewrite_defs_refs_to_components(mut value: Value) -> Value {
    rewrite_refs(&mut value);
    value
}

fn rewrite_refs(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(text)) = map.get_mut("$ref")
                && let Some(name) = text.strip_prefix("#/$defs/")
            {
                *text = format!("#/components/schemas/{name}");
            }
            for child in map.values_mut() {
                rewrite_refs(child);
            }
            // strip $defs / $schema — they live at root in OpenAPI components,
            // not inside each schema body.
            map.remove("$defs");
            map.remove("$schema");
        }
        Value::Array(items) => {
            for item in items {
                rewrite_refs(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::{FieldFlags, StructField};
    use alloc::vec;

    #[test]
    fn emits_openapi_envelope_with_components_schemas() {
        let mut schemas = BTreeMap::new();
        schemas.insert(
            "User".into(),
            Schema::Struct {
                name: "User".into(),
                fields: vec![StructField {
                    name: "name".into(),
                    schema: Schema::String {
                        pattern: None,
                        format: None,
                        min_len: Some(1),
                        max_len: None,
                    },
                    flags: FieldFlags::default(),
                }],
            },
        );
        let document = emit("hello-api", "0.1.0", &schemas, json!({}));
        assert_eq!(document["openapi"], "3.1.0");
        assert_eq!(document["info"]["title"], "hello-api");
        assert_eq!(document["info"]["version"], "0.1.0");
        assert_eq!(
            document["components"]["schemas"]["User"]["properties"]["name"]["type"],
            "string"
        );
        // $defs and $schema must not leak into component bodies.
        assert!(document["components"]["schemas"]["User"]["$defs"].is_null());
        assert!(document["components"]["schemas"]["User"]["$schema"].is_null());
    }

    #[test]
    fn rewrites_defs_refs_to_components_schemas() {
        let mut schemas = BTreeMap::new();
        schemas.insert(
            "Outer".into(),
            Schema::Struct {
                name: "Outer".into(),
                fields: vec![StructField {
                    name: "inner".into(),
                    schema: Schema::Ref("Inner".into()),
                    flags: FieldFlags::default(),
                }],
            },
        );
        schemas.insert("Inner".into(), Schema::Bool);
        let document = emit("api", "0.1.0", &schemas, json!({}));
        let outer = &document["components"]["schemas"]["Outer"];
        assert_eq!(
            outer["properties"]["inner"]["$ref"],
            "#/components/schemas/Inner"
        );
    }
}
