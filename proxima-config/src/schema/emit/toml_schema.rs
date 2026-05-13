//! Emit proxima's self-describing TOML schema form. Every registered
//! schema becomes a `[[schema]]` block matching what `load_full` parses.

use alloc::collections::BTreeMap;

use crate::schema::Schema;
use proxima_core::ProximaError;

/// Serialize a registry of named schemas as the same `[[schema]]`
/// blocks that proxima parses back from a pipe config.
pub fn emit(schemas: &BTreeMap<String, Schema>) -> Result<String, ProximaError> {
    let mut blocks = Vec::with_capacity(schemas.len());
    for (name, schema) in schemas {
        let entry = SchemaBlock {
            name: name.clone(),
            schema: schema.clone(),
        };
        blocks.push(entry);
    }
    let document = SchemaDocument { schema: blocks };
    toml::to_string_pretty(&document)
        .map_err(|err| ProximaError::Config(format!("toml-schema: {err}")))
}

#[derive(serde::Serialize)]
struct SchemaDocument {
    #[serde(rename = "schema")]
    schema: Vec<SchemaBlock>,
}

#[derive(serde::Serialize)]
struct SchemaBlock {
    name: String,
    schema: Schema,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::{FieldFlags, StructField};

    #[test]
    fn round_trips_a_struct_through_toml() {
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
        let text = emit(&schemas).expect("emit");
        assert!(text.contains("[[schema]]"));
        assert!(text.contains("name = \"User\""));
        // re-parse the emitted toml and verify the round trip
        #[derive(serde::Deserialize)]
        struct Doc {
            schema: Vec<Block>,
        }
        #[derive(serde::Deserialize)]
        struct Block {
            name: String,
            schema: Schema,
        }
        let parsed: Doc = toml::from_str(&text).expect("parse");
        assert_eq!(parsed.schema.len(), 1);
        assert_eq!(parsed.schema[0].name, "User");
        match &parsed.schema[0].schema {
            Schema::Struct { fields, .. } => assert_eq!(fields.len(), 1),
            other => panic!("expected struct, got {other:?}"),
        }
    }
}
